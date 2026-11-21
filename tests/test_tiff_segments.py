# test_tiff_segments.py

# Copyright (c) 2008-2026, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Segmented tile/strip access tests for the tifffile package.

These tests exercise the connection between page iteration, tile/strip
decoding (``TiffPage.segments`` / ``TiffFile.read_segments``) and the
whole-image decode paths (``TiffFile.asarray`` / ``TiffPage.asarray``).

All sample images are generated offline inside the tests. Every assertion
compares explicit pixel values, not just array shapes. Both TIFF byte
orders, odd image dimensions, short final segments, row padding, edge
tiles and multiple compressions are covered. Hand-crafted minimal TIFF
files are cross-checked against files written by :py:class:`TiffWriter`.

"""

from __future__ import annotations

import io
import logging
import struct
import warnings
from typing import TYPE_CHECKING, Any

import numpy
import pytest
from numpy import array_equal

import tifffile
from tifffile import (
    COMPRESSION,
    ORIENTATION,
    PHOTOMETRIC,
    SAMPLEFORMAT,
    TiffFile,
    TiffFileError,
    TiffWriter,
    imread,
    imwrite,
)
from tifffile.tifffile import TiffPage, apply_colormap, reorient

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray


try:
    import imagecodecs  # noqa: F401

    HAVE_IMAGECODECS = True
except ImportError:
    # tifffile's bundled fallback only supports deflate/zlib, lzma and zstd
    HAVE_IMAGECODECS = False

# compressions requiring the optional imagecodecs package (LZW, PackBits)
CODEC_COMPRESSIONS = (
    COMPRESSION.PACKBITS,
    COMPRESSION.LZW,
)
needs_imagecodecs = pytest.mark.skipif(
    not HAVE_IMAGECODECS,
    reason='compression requires the imagecodecs package',
)


BYTEORDERS = ('<', '>')
# compressions available without optional codecs and with tifffile's
# bundled pure-Python fallback as well as imagecodecs
COMPRESSIONS = (
    None,
    COMPRESSION.PACKBITS,
    COMPRESSION.ADOBE_DEFLATE,
    COMPRESSION.LZW,
)


###############################################################################
# Helpers and offline, hand-crafted TIFF samples
###############################################################################


def pattern(shape: tuple[int, ...], dtype: DTypeLike) -> NDArray[Any]:
    """Return deterministic array with unique, bounded pixel values."""
    dtype = numpy.dtype(dtype)
    size = int(numpy.prod(shape))
    if numpy.issubdtype(dtype, numpy.floating):
        data = numpy.arange(size, dtype=numpy.float64).reshape(shape)
        return data.astype(dtype)
    info = numpy.iinfo(dtype)
    # spread values so adjacent pixels differ, including high bytes
    values = (numpy.arange(size, dtype=numpy.int64) * 257 + 11) % (
        int(info.max) + 1
    )
    if info.min < 0:
        values = values - (int(info.max) + 1) // 2
    return values.astype(dtype).reshape(shape)


def assemble_segments(
    page: TiffPage, fillvalue: float | None = None
) -> NDArray[Any]:
    """Decode every segment of a page individually and place it by index.

    Mirrors what ``TiffPage.asarray`` does internally, but goes through the
    public ``TiffPage.segments`` iterator one segment at a time.

    """
    kwargs = {} if fillvalue is None else {'fillvalue': fillvalue}
    result = numpy.empty(page.shaped, dtype=page.dtype)
    for segment, position, segment_shape in page.segments(**kwargs):
        sample, depth, length, width, _contig = position
        d_size = min(segment_shape[0], page.imagedepth - depth)
        h_size = min(segment_shape[1], page.imagelength - length)
        w_size = min(segment_shape[2], page.imagewidth - width)
        if segment is None:
            # empty segment: fillvalue is applied by asarray, not segments
            value = page.fillvalue if fillvalue is None else fillvalue
            result[
                sample,
                depth : depth + d_size,
                length : length + h_size,
                width : width + w_size,
            ] = value
        else:
            result[
                sample,
                depth : depth + d_size,
                length : length + h_size,
                width : width + w_size,
            ] = segment[:d_size, :h_size, :w_size]
    return result.reshape(page.shape)


def write_tag(
    buffer: io.BytesIO,
    byteorder: str,
    code: int,
    dtype: int,
    count: int,
    value: int,
    *,
    is_offset: bool = False,
) -> None:
    """Write one classic (12-byte) TIFF tag entry with an inline value.

    Set ``is_offset`` for out-of-line values: the four-byte value field then
    holds a LONG offset regardless of the tag's element data type.

    """
    if is_offset:
        packed = struct.pack(byteorder + 'I', value)
    else:
        fmt = {1: 'B', 2: 'B', 3: 'H', 4: 'I'}[dtype]
        packed = struct.pack(byteorder + fmt, value)
    buffer.write(struct.pack(byteorder + 'HHI', code, dtype, count))
    if is_offset or len(packed) == 4:
        # LONG (or an offset field) occupies all four value bytes
        buffer.write(packed)
    else:
        # shorter inline values are left-aligned (lowest address)
        buffer.write(packed + b'\x00' * (4 - len(packed)))


def build_minimal_tiff(
    byteorder: str,
    arrays: list[NDArray[Any]],
    *,
    photometric: int = PHOTOMETRIC.MINISBLACK,
    colormap: NDArray[Any] | None = None,
    orientation: int | None = None,
) -> bytes:
    """Build a minimal, uncompressed, striped classic TIFF from raw parts.

    The layout is computed explicitly so the test does not depend on
    :py:class:`TiffWriter` for the file under test. Multi-page IFD chains,
    both byte orders, palette and orientation tags are supported.

    """
    bo = byteorder
    # colormap values are stored out-of-line; its tag entry holds an offset
    extra = 1 if colormap is not None else 0
    if orientation is not None:
        extra += 1
    ntags = 9 + extra
    ifd_size = 2 + 12 * ntags + 4

    ifd_offsets: list[int] = []
    data_offsets: list[int] = []
    cursor = 8
    blobs: list[bytes] = []
    for array in arrays:
        ifd_offsets.append(cursor)
        cursor += ifd_size
    for array in arrays:
        blob = array.astype(bo + array.dtype.char).tobytes()
        data_offsets.append(cursor)
        blobs.append(blob)
        cursor += len(blob)
    colormap_offset = cursor if colormap is not None else 0
    colormap_bytes = (
        colormap.astype(bo + 'u2').tobytes() if colormap is not None else b''
    )
    cursor += len(colormap_bytes)

    out = io.BytesIO()
    out.write(b'II' if bo == '<' else b'MM')
    out.write(struct.pack(bo + 'H', 42))
    out.write(struct.pack(bo + 'I', ifd_offsets[0]))

    for index, (array, data_offset) in enumerate(
        zip(arrays, data_offsets, strict=True)
    ):
        image_length, image_width = array.shape
        bits = array.dtype.itemsize * 8
        tags: list[tuple[int, int, int, int]] = [
            (256, 4, 1, image_width),
            (257, 4, 1, image_length),
            (258, 3, 1, bits),
            (259, 3, 1, COMPRESSION.NONE),
            (262, 3, 1, photometric),
            (273, 4, 1, data_offset),
            (277, 3, 1, 1),
            (278, 4, 1, image_length),
            (279, 4, 1, array.nbytes),
        ]
        if colormap is not None:
            tags.append((320, 3, colormap.size, colormap_offset))
        if orientation is not None:
            tags.append((274, 3, 1, orientation))
        tags.sort()

        out.seek(ifd_offsets[index])
        out.write(struct.pack(bo + 'H', len(tags)))
        for code, dtype, count, value in tags:
            # values larger than four bytes are stored out-of-line and the
            # tag field holds a LONG offset (e.g. the Colormap tag)
            write_tag(
                out, bo, code, dtype, count, value,
                is_offset=dtype * count > 4,
            )
        next_ifd = ifd_offsets[index + 1] if index + 1 < len(arrays) else 0
        out.write(struct.pack(bo + 'I', next_ifd))

    for data_offset, blob in zip(data_offsets, blobs, strict=True):
        out.seek(data_offset)
        out.write(blob)
    if colormap is not None:
        out.seek(colormap_offset)
        out.write(colormap_bytes)
    return out.getvalue()


def patch_bytes(
    raw: bytes, offset: int, data: bytes
) -> bytes:
    """Return copy of raw TIFF bytes with data overwritten at offset."""
    patched = bytearray(raw)
    patched[offset : offset + len(data)] = data
    return bytes(patched)


@pytest.fixture
def tmp_file(tmp_path: Any) -> Any:
    """Return factory for temporary TIFF file paths."""

    def factory(name: str) -> str:
        return str(tmp_path / name)

    return factory


###############################################################################
# Hand-crafted minimal TIFF vs TiffWriter output
###############################################################################


@pytest.mark.parametrize('byteorder', BYTEORDERS)
def test_manual_minimal_tiff_matches_writer(
    byteorder: str, tmp_file: Any
) -> None:
    """Hand-crafted single-page striped TIFF decodes to exact pixels."""
    data = pattern((3, 5), numpy.uint16)
    raw = build_minimal_tiff(byteorder, [data])
    filename = tmp_file(f'manual_strip_{byteorder}.tif')
    with open(filename, 'wb') as fh:
        fh.write(raw)

    with TiffFile(filename) as tif:
        assert tif.byteorder == byteorder
        assert len(tif.pages) == 1
        page = tif.pages.first
        assert page.is_tiled is False
        assert page.imagewidth == 5
        assert page.imagelength == 3
        assert page.bitspersample == 16
        assert page.sampleformat == SAMPLEFORMAT.UINT
        assert page.compression == COMPRESSION.NONE
        # full decode through TiffFile and page entry points
        assert array_equal(tif.asarray(), data)
        assert array_equal(page.asarray(), data)
        # segmented decode must reproduce every pixel
        assert array_equal(assemble_segments(page), data)
        # module-level entry point
    assert array_equal(imread(filename), data)

    # cross-check against TiffWriter output with the same byte order
    reference = tmp_file(f'ref_strip_{byteorder}.tif')
    imwrite(reference, data, byteorder=byteorder)
    with TiffFile(reference) as tif:
        assert tif.byteorder == byteorder
        assert array_equal(tif.asarray(), data)


@pytest.mark.parametrize('byteorder', BYTEORDERS)
def test_manual_multipage_iteration(
    byteorder: str, tmp_file: Any
) -> None:
    """IFD chain of hand-crafted pages iterates and decodes independently."""
    page0 = pattern((3, 5), numpy.uint8)
    page1 = pattern((2, 4), numpy.uint8)
    page2 = pattern((4, 2), numpy.uint8)
    raw = build_minimal_tiff(byteorder, [page0, page1, page2])
    filename = tmp_file(f'manual_pages_{byteorder}.tif')
    with open(filename, 'wb') as fh:
        fh.write(raw)

    with TiffFile(filename) as tif:
        assert tif.byteorder == byteorder
        assert len(tif.pages) == 3
        # iteration follows the next-IFD chain
        decoded = [page.asarray() for page in tif.pages]
        assert [array.shape for array in decoded] == [
            (3, 5),
            (2, 4),
            (4, 2),
        ]
        assert array_equal(decoded[0], page0)
        assert array_equal(decoded[1], page1)
        assert array_equal(decoded[2], page2)
        # indexed access and iteration expose the same page objects' data
        for index, expected in enumerate((page0, page1, page2)):
            assert array_equal(tif.pages[index].asarray(), expected)
        # each page owns its strip offsets/byte counts
        offsets = [page.dataoffsets[0] for page in tif.pages]
        assert offsets == sorted(offsets)
        assert len({offsets[0]}) == 1 or offsets[0] != offsets[1]


@pytest.mark.parametrize('byteorder', BYTEORDERS)
def test_manual_palette_tiff(
    byteorder: str, tmp_file: Any
) -> None:
    """Palette tag, index pixels and applied colormap correspond exactly."""
    indices = numpy.array(
        [[0, 1, 2, 7], [255, 128, 3, 9]], dtype=numpy.uint8
    )
    ramp = numpy.arange(256, dtype=numpy.uint32)
    colormap = numpy.stack(
        [
            ramp.astype(numpy.uint16),
            ((ramp * 257) % 65536).astype(numpy.uint16),
            (65535 - ramp).astype(numpy.uint16),
        ]
    )
    raw = build_minimal_tiff(
        byteorder,
        [indices],
        photometric=PHOTOMETRIC.PALETTE,
        colormap=colormap,
    )
    filename = tmp_file(f'manual_palette_{byteorder}.tif')
    with open(filename, 'wb') as fh:
        fh.write(raw)

    with TiffFile(filename) as tif:
        page = tif.pages.first
        assert int(page.photometric) == PHOTOMETRIC.PALETTE
        assert page.bitspersample == 8
        # raw decode returns palette indices, never expanded RGB
        raw_image = page.asarray()
        assert raw_image.dtype == numpy.uint8
        assert array_equal(raw_image, indices)
        # tag values match the array used as palette
        assert page.colormap is not None
        assert page.colormap.dtype == numpy.uint16
        assert array_equal(page.colormap, colormap)
        # applying the palette yields the documented RGB mapping
        expected_rgb = apply_colormap(indices, colormap)
        assert array_equal(page.asrgb(), expected_rgb)
        # spot-check explicit pixel coordinates
        assert tuple(page.asrgb()[0, 3]) == tuple(
            colormap[:, indices[0, 3]]
        )
        assert tuple(page.asrgb()[1, 0]) == tuple(
            colormap[:, indices[1, 0]]
        )


@pytest.mark.parametrize('byteorder', BYTEORDERS)
def test_manual_orientation_tag(
    byteorder: str, tmp_file: Any
) -> None:
    """Orientation tag is exposed; raw pixels stay in stored order."""
    data = pattern((4, 3), numpy.uint8)
    raw = build_minimal_tiff(
        byteorder, [data], orientation=ORIENTATION.BOTRIGHT
    )
    filename = tmp_file(f'manual_orientation_{byteorder}.tif')
    with open(filename, 'wb') as fh:
        fh.write(raw)

    with TiffFile(filename) as tif:
        page = tif.pages.first
        assert page.tags[274].value == ORIENTATION.BOTRIGHT
        # asarray does not reorient: stored pixel order is preserved
        stored = page.asarray()
        assert array_equal(stored, data)
        # orientation can be applied explicitly to the unsqueezed array and
        # matches the expected 180-degree rotation of the stored pixels
        unsqueezed = page.asarray(squeeze=False)
        assert unsqueezed.shape[-3:-1] == data.shape
        reoriented = reorient(
            unsqueezed, ORIENTATION.BOTRIGHT
        ).reshape(data.shape)
        assert array_equal(reoriented, data[::-1, ::-1])
        assert array_equal(reoriented, stored[::-1, ::-1])


###############################################################################
# Tile decoding: odd dimensions, edge tiles, byte orders and compressions
###############################################################################


def _compression_params() -> list[Any]:
    """Parametrize compressions, skipping codecs that are unavailable."""
    params: list[Any] = []
    for compression in COMPRESSIONS:
        marks = (
            [needs_imagecodecs]
            if compression in CODEC_COMPRESSIONS
            else []
        )
        params.append(
            pytest.param(compression, marks=marks, id=str(compression))
        )
    return params


def _compressed_params() -> list[Any]:
    """Parametrize only actual compressions (excluding uncompressed)."""
    return [
        param
        for param in _compression_params()
        if param.values[0] is not None
    ]


@pytest.mark.parametrize('byteorder', BYTEORDERS)
@pytest.mark.parametrize('compression', _compression_params())
def test_tiled_odd_dimensions_per_pixel(
    byteorder: str, compression: int | None, tmp_file: Any
) -> None:
    """Tiled image with partial edge tiles decodes pixel for pixel.

    A 20x18 image stored in 16x16 tiles yields four tiles; the right and
    bottom edge tiles cover fewer pixels than the full tile size.

    """
    data = pattern((20, 18), numpy.uint16)
    filename = tmp_file(f'tiles_{byteorder}_{compression}.tif')
    kwargs = {} if compression is None else {'compression': compression}
    imwrite(filename, data, tile=(16, 16), byteorder=byteorder, **kwargs)

    with TiffFile(filename) as tif:
        assert tif.byteorder == byteorder
        page = tif.pages.first
        assert page.is_tiled
        assert (page.tilelength, page.tilewidth) == (16, 16)
        assert page.chunked == (2, 2)
        assert page.bitspersample == 16
        assert page.dtype == numpy.dtype(numpy.uint16)

        segments = list(page.segments())
        assert len(segments) == 4
        expected_positions = [
            (0, 0, 0, 0, 0),
            (0, 0, 0, 16, 0),
            (0, 0, 16, 0, 0),
            (0, 0, 16, 16, 0),
        ]
        assert [item[1] for item in segments] == expected_positions
        # full-size tiles are always returned; the image-sized region is
        # checked against the source, including every edge pixel
        for segment, position, shape in segments:
            assert segment is not None
            assert shape == (1, 16, 16, 1)
            _, _, h, w, _ = position
            edge_h = min(16, data.shape[0] - h)
            edge_w = min(16, data.shape[1] - w)
            assert array_equal(
                segment[0, :edge_h, :edge_w, 0],
                data[h : h + edge_h, w : w + edge_w],
            )

        # assembling segments through the public iterator matches asarray
        assembled = assemble_segments(page)
        full = page.asarray()
        assert array_equal(assembled, data)
        assert array_equal(full, data)
        assert array_equal(tif.asarray(), data)

        # explicit edge coordinates must not leak adjacent tile rows/cols
        assert full[15, 17] == data[15, 17]
        assert full[16, 0] == data[16, 0]
        assert full[19, 17] == data[19, 17]
        # padding area of edge tiles is not part of the decoded image
        assert full.shape == data.shape


@pytest.mark.parametrize('byteorder', BYTEORDERS)
def test_tiled_edge_tiles_no_neighbor_leak(
    byteorder: str, tmp_file: Any
) -> None:
    """Distinct tile values prove edge pixels cannot come from neighbors."""
    # four 16x16 logical tiles with unique, non-repeating value bands
    data = numpy.empty((20, 18), dtype=numpy.uint16)
    data[:16, :16] = 1
    data[:16, 16:] = 2
    data[16:, :16] = 3
    data[16:, 16:] = 4
    filename = tmp_file(f'tile_leak_{byteorder}.tif')
    imwrite(filename, data, tile=(16, 16), byteorder=byteorder)

    with TiffFile(filename) as tif:
        page = tif.pages.first
        segments = list(page.segments())
        # bottom-right tile's valid region is 4x2 and must contain only 4s
        last_tile = segments[-1][0]
        assert (last_tile[0, :4, :2, 0] == 4).all()
        # the padded part of the partial tile is zero-filled
        assert (last_tile[0, 4:, :, 0] == 0).all()
        assert (last_tile[0, :, 2:, 0] == 0).all()
        full = page.asarray()
        assert set(numpy.unique(full).tolist()) == {1, 2, 3, 4}
        assert (full[16:, 16:] == 4).all()
        assert (full[16:, :16] == 3).all()
        assert (full[:16, 16:] == 2).all()


###############################################################################
# Strip decoding: row padding, short final strip and explicit pixel checks
###############################################################################


@pytest.mark.parametrize('byteorder', BYTEORDERS)
@pytest.mark.parametrize('compression', _compression_params())
def test_stripped_odd_dimensions_per_pixel(
    byteorder: str, compression: int | None, tmp_file: Any
) -> None:
    """Strips with rowsperstrip not dividing image length decode exactly."""
    data = pattern((5, 7), numpy.uint16)
    filename = tmp_file(f'strips_{byteorder}_{compression}.tif')
    kwargs = {} if compression is None else {'compression': compression}
    imwrite(
        filename,
        data,
        rowsperstrip=2,
        byteorder=byteorder,
        **kwargs,
    )

    with TiffFile(filename) as tif:
        assert tif.byteorder == byteorder
        page = tif.pages.first
        assert not page.is_tiled
        assert page.rowsperstrip == 2
        assert page.chunked == (3, 1)
        assert len(page.dataoffsets) == 3
        # byte counts reflect row padding: full rows are 14 bytes, last
        # strip holds a single row; compressed counts are checked only by
        # number of strips because their sizes depend on the codec
        assert len(page.databytecounts) == 3
        if compression is None:
            assert page.databytecounts == (28, 28, 14)

        segments = list(page.segments())
        assert len(segments) == 3
        row_ranges = [(0, 2), (2, 4), (4, 5)]
        for (segment, _position, shape), (row0, row1) in zip(
            segments, row_ranges, strict=True
        ):
            assert segment is not None
            assert shape[1] == row1 - row0  # last strip has one row
            assert array_equal(
                segment.reshape(row1 - row0, 7), data[row0:row1]
            )

        assert array_equal(assemble_segments(page), data)
        assert array_equal(page.asarray(), data)
        assert array_equal(tif.asarray(), data)
        # last-row pixel values explicitly, including odd-width boundary
        assert page.asarray()[4].tolist() == data[4].tolist()
        assert page.asarray()[4, 6] == data[4, 6]


def test_strip_row_padding_8bit(tmp_file: Any) -> None:
    """Odd 8-bit width produces no implicit row padding in strip bytes."""
    data = pattern((5, 3), numpy.uint8)
    filename = tmp_file('strip_rowpad.tif')
    imwrite(filename, data, rowsperstrip=2)
    with TiffFile(filename) as tif:
        page = tif.pages.first
        # 3 samples per row, no padding bytes
        assert page.databytecounts == (6, 6, 3)
        for (segment, _pos, _shape), (row0, row1) in zip(
            page.segments(), [(0, 2), (2, 4), (4, 5)], strict=True
        ):
            nrows = row1 - row0
            assert array_equal(
                segment.reshape(nrows, 3), data[row0:row1]
            )
        assert array_equal(page.asarray(), data)


###############################################################################
# 16-bit depth and sample format preservation
###############################################################################


@pytest.mark.parametrize('byteorder', BYTEORDERS)
def test_uint16_grayscale_not_downcast(
    byteorder: str, tmp_file: Any
) -> None:
    """16-bit grayscale survives both entry points without downscaling."""
    data = numpy.array(
        [[0, 1, 255, 256], [257, 1024, 32768, 65535]], dtype=numpy.uint16
    )
    filename = tmp_file(f'gray16_{byteorder}.tif')
    imwrite(filename, data, byteorder=byteorder)

    with TiffFile(filename) as tif:
        page = tif.pages.first
        assert page.bitspersample == 16
        assert int(page.sampleformat) == SAMPLEFORMAT.UINT
        # decoded dtype is native-endian uint16 regardless of file endian
        assert page.dtype == numpy.dtype(numpy.uint16)
        image = tif.asarray()
        assert image.dtype == numpy.uint16  # native dtype, full 16-bit range
        assert array_equal(image, data)
        assert int(image.max()) == 65535
        assert int(image[0, 3]) == 256
        # the non-native file order only shows in the on-disk dtype metadata
        assert page.parent.byteorder == byteorder
        # segment path preserves dtype too
        for segment, _pos, _shape in page.segments():
            assert segment.dtype == numpy.uint16


@pytest.mark.parametrize('byteorder', BYTEORDERS)
def test_int16_signed_sampleformat(
    byteorder: str, tmp_file: Any
) -> None:
    """Signed integer sample format and negative values are preserved."""
    data = numpy.array([[-1, 0, 1], [-32768, 32767, 12]], dtype=numpy.int16)
    filename = tmp_file(f'int16_{byteorder}.tif')
    imwrite(filename, data, byteorder=byteorder)
    with TiffFile(filename) as tif:
        page = tif.pages.first
        assert page.bitspersample == 16
        assert int(page.sampleformat) == SAMPLEFORMAT.INT
        image = page.asarray()
        assert image.dtype == numpy.int16
        assert array_equal(image, data)
        assert array_equal(assemble_segments(page), data)


###############################################################################
# Truncated final segments and fillvalue
###############################################################################


def test_truncated_final_tile_fillvalue(tmp_file: Any) -> None:
    """Short read of final uncompressed tile is padded per fillvalue."""
    data = pattern((20, 18), numpy.uint8)
    source = tmp_file('trunc_source.tif')
    imwrite(source, data, tile=(16, 16))
    with TiffFile(source) as tif:
        page = tif.pages.first
        last_offset = page.dataoffsets[-1]
        assert page.databytecounts[-1] == 256
        assert not page.is_contiguous  # multi-tile: segment decode path

    # physically truncate the file, leaving only six bytes of the last tile
    raw = open(source, 'rb').read()
    cut = raw[: last_offset + 6]
    filename = tmp_file('truncated_tile.tif')
    with open(filename, 'wb') as fh:
        fh.write(cut)

    with TiffFile(filename) as tif:
        page = tif.pages.first
        assert page.databytecounts[-1] == 256  # tag unchanged
        # the partial edge tile covers rows 16:20, cols 16:18 (4x2 pixels)
        edge = data[16:, 16:]

        # Decode order: the affected tile is first expanded to a full
        # 16x16 buffer, where a six-byte short read leaves six valid samples
        # followed by fillvalue. Only the first two valid samples fall
        # inside the cropped edge region (its first row); the rest of that
        # region is padding. The other three tiles decode untouched.
        default = page.asarray()
        assert array_equal(default[:16], data[:16])
        assert array_equal(default[16:, :16], data[16:, :16])  # bottom-left
        assert array_equal(default[16, 16:18], edge[0, :2])
        assert default[16, 18:].shape == (0,)  # image width is 18
        assert (default[17:, 16:] == 0).all()

        # explicit fillvalue through both asarray and the segment iterator
        filled = page.asarray(fillvalue=9)
        assert array_equal(filled[:16], data[:16])
        assert array_equal(filled[16:, :16], data[16:, :16])
        assert array_equal(filled[16, 16:18], edge[0, :2])
        assert (filled[17:, 16:] == 9).all()
        assembled = assemble_segments(page, fillvalue=9)
        assert array_equal(assembled, filled)

        # the uncropped tile buffer exposes the linear fillvalue padding.
        # its first six samples come from the start of the full-size tile
        # buffer (the image's row 16, cols 16 and 17, then padding area),
        # followed by 250 fillvalue samples
        full_tile = numpy.zeros((16, 16), dtype=numpy.uint8)
        full_tile[: data.shape[0] - 16, : data.shape[1] - 16] = data[16:, 16:]
        last_segment = list(page.segments(fillvalue=9))[-1][0]
        assert last_segment is not None
        assert last_segment.shape == (1, 16, 16, 1)
        assert array_equal(last_segment.flatten()[:6], full_tile.flatten()[:6])
        assert (last_segment.flatten()[6:] == 9).all()
        default_segment = list(page.segments())[-1][0]
        assert array_equal(
            default_segment.flatten()[:6], full_tile.flatten()[:6]
        )
        assert (default_segment.flatten()[6:] == 0).all()


def build_gapped_strip_tiff(
    data: NDArray[Any], rowsperstrip: int, gap: bytes = b'\xde\xad\xbe\xef'
) -> bytes:
    """Build uncompressed multi-strip TIFF with gaps between strips.

    TiffWriter lays uncompressed strips out contiguously, which takes the
    fast ``read_array`` path. Inserting explicit gaps forces the segmented
    read/decode path while keeping the strips themselves uncompressed.

    StripOffsets and StripByteCounts are stored as out-of-line arrays.

    """
    assert data.ndim == 2
    image_length, image_width = data.shape
    rows = [
        data[row : row + rowsperstrip]
        for row in range(0, image_length, rowsperstrip)
    ]
    blobs = [row.tobytes() for row in rows]
    ntags = 9
    ifd_size = 2 + 12 * ntags + 4
    offsets_array = 8 + ifd_size
    counts_array = offsets_array + 4 * len(blobs)
    cursor = counts_array + 4 * len(blobs)
    strip_offsets = []
    for blob in blobs:
        strip_offsets.append(cursor)
        cursor += len(blob) + len(gap)
    strip_counts = [len(blob) for blob in blobs]

    tags = [
        (256, 4, 1, image_width, 'I'),
        (257, 4, 1, image_length, 'I'),
        (258, 3, 1, data.dtype.itemsize * 8, 'H'),
        (259, 3, 1, COMPRESSION.NONE, 'H'),
        (262, 3, 1, PHOTOMETRIC.MINISBLACK, 'H'),
        (273, 4, len(blobs), offsets_array, 'I'),
        (277, 3, 1, 1, 'H'),
        (278, 4, 1, rowsperstrip, 'I'),
        (279, 4, len(blobs), counts_array, 'I'),
    ]
    tags.sort()
    out = io.BytesIO()
    out.write(b'II')
    out.write(struct.pack('<H', 42))
    out.write(struct.pack('<I', 8))
    out.seek(8)
    out.write(struct.pack('<H', len(tags)))
    for code, dtype, _count, value, fmt in tags:
        write_tag(
            out, '<', code, dtype, _count, value,
            is_offset=dtype * _count > 4,
        )
    out.write(struct.pack('<I', 0))
    out.seek(offsets_array)
    out.write(struct.pack(f'<{len(blobs)}I', *strip_offsets))
    out.seek(counts_array)
    out.write(struct.pack(f'<{len(blobs)}I', *strip_counts))
    for offset, blob in zip(strip_offsets, blobs, strict=True):
        out.seek(offset)
        out.write(blob)
        out.write(gap)
    return out.getvalue()


def test_truncated_final_strip_fillvalue(tmp_file: Any) -> None:
    """Short read of final uncompressed strip pads with fillvalue."""
    data = pattern((5, 4), numpy.uint8)
    raw = build_gapped_strip_tiff(data, rowsperstrip=2)
    source = tmp_file('gapped_strips.tif')
    with open(source, 'wb') as fh:
        fh.write(raw)

    with TiffFile(source) as tif:
        page = tif.pages.first
        assert not page.is_contiguous  # gaps force the segment path
        assert page.rowsperstrip == 2
        assert page.databytecounts == (8, 8, 4)
        assert array_equal(page.asarray(), data)
        last_offset = page.dataoffsets[-1]

    # keep only two of the four bytes of the final strip
    filename = tmp_file('gapped_strips_truncated.tif')
    with open(filename, 'wb') as fh:
        fh.write(raw[: last_offset + 2])

    with TiffFile(filename) as tif:
        page = tif.pages.first
        image = page.asarray(fillvalue=77)
        assert array_equal(image[:4], data[:4])
        assert image[4, :2].tolist() == data[4, :2].tolist()
        assert (image[4, 2:] == 77).all()
        assembled = assemble_segments(page, fillvalue=77)
        assert array_equal(assembled, image)
        # default fillvalue is zero
        default = page.asarray()
        assert array_equal(default[:4], data[:4])
        assert (default[4, 2:] == 0).all()


def test_empty_segment_fillvalue(tmp_file: Any) -> None:
    """Sparse segment with zero offset/byte count is filled, not skipped."""
    data = pattern((20, 18), numpy.uint8)
    source = tmp_file('empty_tile_source.tif')
    imwrite(source, data, tile=(16, 16))
    with TiffFile(source) as tif:
        page = tif.pages.first
        offsets_tag = page.tags[324]
        counts_tag = page.tags[325]
        assert offsets_tag.count == 4

    raw = bytearray(open(source, 'rb').read())
    # tile indices are column-major across the tile grid: 0 top-left,
    # 1 top-right, 2 bottom-left, 3 bottom-right (see TiffPage.indices).
    # Mark the top-right tile (index 1) as missing. TileOffsets and
    # TileByteCounts may use different TIFF data types and therefore have
    # different per-element sizes.
    offset_format = tifffile.TIFF.DATA_FORMATS[int(offsets_tag.dtype)]
    count_format = tifffile.TIFF.DATA_FORMATS[int(counts_tag.dtype)]
    offset_itemsize = struct.calcsize(offset_format)
    count_itemsize = struct.calcsize(count_format)
    struct.pack_into(
        '<' + offset_format,
        raw,
        offsets_tag.valueoffset + offset_itemsize,
        0,
    )
    struct.pack_into(
        '<' + count_format,
        raw,
        counts_tag.valueoffset + count_itemsize,
        0,
    )
    filename = tmp_file('empty_tile.tif')
    with open(filename, 'wb') as fh:
        fh.write(bytes(raw))

    with TiffFile(filename) as tif:
        page = tif.pages.first
        assert page.dataoffsets[1] == 0
        assert page.databytecounts[1] == 0
        assert all(offset > 0 for offset in (*page.dataoffsets[:1],
                                             *page.dataoffsets[2:]))
        assert all(count == 256 for count in (*page.databytecounts[:1],
                                              *page.databytecounts[2:]))
        segments = list(page.segments())
        assert segments[1][0] is None  # empty segment reported explicitly
        assert segments[1][1] == (0, 0, 0, 16, 0)  # positioned top-right
        image = page.asarray(fillvalue=5)
        # the top-right tile region (rows 0:16, cols 16:18) is filled
        assert (image[:16, 16:] == 5).all()
        assert array_equal(image[:16, :16], data[:16, :16])
        assert array_equal(image[16:], data[16:])
        assembled = assemble_segments(page, fillvalue=5)
        assert array_equal(assembled, image)


def test_contiguous_short_read_raises(tmp_file: Any) -> None:
    """Single contiguous segment short read fails loudly, not silently."""
    data = pattern((16, 16), numpy.uint8)
    source = tmp_file('contig_source.tif')
    imwrite(source, data, tile=(16, 16))
    with TiffFile(source) as tif:
        assert tif.pages.first.is_contiguous
        offset = tif.pages.first.dataoffsets[0]
    raw = open(source, 'rb').read()
    filename = tmp_file('contig_truncated.tif')
    with open(filename, 'wb') as fh:
        fh.write(raw[: offset + 8])

    with TiffFile(filename) as tif:
        with pytest.raises(ValueError, match='failed to read'):
            tif.asarray()


@pytest.mark.parametrize('compression', _compressed_params())
def test_corrupted_compressed_segment_raises_runtimeerror(
    compression: int, tmp_file: Any
) -> None:
    """Damaged compressed streams raise a RuntimeError, not bad pixels."""
    data = pattern((6, 4), numpy.uint8)
    source = tmp_file(f'bad_comp_{int(compression)}.tif')
    imwrite(source, data, rowsperstrip=2, compression=compression)
    with TiffFile(source) as tif:
        page = tif.pages.first
        offset = page.dataoffsets[0]
        count = page.databytecounts[0]
    raw = bytearray(open(source, 'rb').read())
    middle = offset + count // 2
    raw[middle] ^= 0xFF
    raw[middle + 1] ^= 0xAA
    filename = tmp_file(f'bad_comp_broken_{int(compression)}.tif')
    with open(filename, 'wb') as fh:
        fh.write(bytes(raw))
    # imagecodecs raises its own RuntimeError subclasses; tifffile's
    # pure-Python deflate fallback raises zlib.error instead
    expected_error: tuple[type[BaseException], ...]
    if HAVE_IMAGECODECS:
        expected_error = (RuntimeError,)
    else:
        import zlib

        expected_error = (zlib.error,)
    with TiffFile(filename) as tif:
        with pytest.raises(expected_error):
            tif.asarray()
        with pytest.raises(expected_error):
            list(tif.pages.first.segments())


###############################################################################
# Corrupt tags and structure: explicit failure types by stage
###############################################################################


def test_corrupt_header_raises(tmp_file: Any) -> None:
    """Invalid byte-order magic fails while opening the TiffFile."""
    source = tmp_file('header_source.tif')
    imwrite(source, pattern((4, 4), numpy.uint8))
    raw = bytearray(open(source, 'rb').read())
    raw[0:2] = b'XX'
    filename = tmp_file('bad_header.tif')
    with open(filename, 'wb') as fh:
        fh.write(bytes(raw))
    with pytest.raises(TiffFileError, match='not a TIFF file'):
        TiffFile(filename)


def test_corrupt_tag_count_raises(tmp_file: Any) -> None:
    """Suspicious IFD tag count and truncated IFD both fail at parse time."""
    source = tmp_file('tagno_source.tif')
    imwrite(source, pattern((4, 4), numpy.uint8))
    with TiffFile(source) as tif:
        ifd_offset = tif.pages.first.offset

    raw = open(source, 'rb').read()
    # implausibly large tag count
    patched = patch_bytes(
        raw, ifd_offset, struct.pack('<H', 9999)
    )
    filename = tmp_file('bad_tagno.tif')
    with open(filename, 'wb') as fh:
        fh.write(patched)
    with pytest.raises(TiffFileError, match='suspicious number of tags'):
        TiffFile(filename)

    # truncated IFD: tag array ends inside the file
    trunc_name = tmp_file('trunc_ifd.tif')
    with open(trunc_name, 'wb') as fh:
        fh.write(raw[: ifd_offset + 4])
    with pytest.raises(TiffFileError, match='corrupted IFD structure'):
        TiffFile(trunc_name)


def test_corrupt_structural_tag_is_rejected(
    tmp_file: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """Invalid data type of a structural tag is a TiffFileError on parse.

    TiffPage catches the per-tag error while reading the IFD and logs it;
    the tag is then absent from the page. The explicit failure type is
    still :class:`TiffFileError` from :py:meth:`TiffTag.fromfile`.

    """
    from tifffile import TiffTag

    source = tmp_file('badtag_source.tif')
    imwrite(source, pattern((4, 4), numpy.uint8))
    source_bytes = open(source, 'rb').read()
    with TiffFile(source) as tif:
        tag = tif.pages.first.tags[258]  # BitsPerSample
        # valueoffset points at the entry's value field (entry start + 8);
        # the data-type field sits two bytes before it
        value_field = tag.valueoffset
        entry_start = value_field - 8
        assert struct.unpack_from(
            '<H', source_bytes, entry_start
        )[0] == 258
        assert struct.unpack_from(
            '<H', source_bytes, entry_start + 2
        )[0] == 3

    raw = bytearray(source_bytes)
    struct.pack_into('<H', raw, entry_start + 2, 99)  # invalid data type
    filename = tmp_file('bad_tag_dtype.tif')
    with open(filename, 'wb') as fh:
        fh.write(bytes(raw))

    with caplog.at_level(logging.ERROR):
        with TiffFile(filename) as tif:
            page = tif.pages.first
            assert 258 not in page.tags
    assert 'invalid data type' in caplog.text

    # the same malformed entry raises TiffFileError through the tag reader
    with TiffFile(filename) as tif:
        with pytest.raises(TiffFileError, match='invalid data type'):
            TiffTag.fromfile(tif, offset=entry_start)


@pytest.mark.parametrize('byteorder', BYTEORDERS)
def test_corrupt_structure_both_endians(
    byteorder: str, tmp_file: Any
) -> None:
    """Parse-stage failures are identical for little and big endian files."""
    source = tmp_file(f'struct_source_{byteorder}.tif')
    imwrite(source, pattern((4, 4), numpy.uint8), byteorder=byteorder)
    raw = open(source, 'rb').read()

    bad_magic = tmp_file(f'struct_magic_{byteorder}.tif')
    with open(bad_magic, 'wb') as fh:
        fh.write(b'XX' + raw[2:])
    with pytest.raises(TiffFileError):
        TiffFile(bad_magic)

    with TiffFile(source) as tif:
        ifd_offset = tif.pages.first.offset
    bad_tagno = tmp_file(f'struct_tagno_{byteorder}.tif')
    with open(bad_tagno, 'wb') as fh:
        fh.write(
            patch_bytes(raw, ifd_offset, struct.pack(byteorder + 'H', 9999))
        )
    with pytest.raises(TiffFileError, match='suspicious number of tags'):
        TiffFile(bad_tagno)


###############################################################################
# read_segments: raw byte access path and short-read semantics
###############################################################################


def test_read_segments_indices_and_bytes(tmp_file: Any) -> None:
    """FileHandle.read_segments returns raw bytes in segment index order."""
    data = pattern((20, 18), numpy.uint8)
    filename = tmp_file('read_segments.tif')
    imwrite(filename, data, tile=(16, 16))

    with TiffFile(filename) as tif:
        page = tif.pages.first
        handle = tif.filehandle
        segments = list(
            handle.read_segments(page.dataoffsets, page.databytecounts)
        )
        assert [index for _blob, index in segments] == [0, 1, 2, 3]
        for (blob, index), offset in zip(
            segments, page.dataoffsets, strict=True
        ):
            assert blob is not None
            handle.seek(offset)
            assert blob == handle.read(page.databytecounts[index])

        # explicit subset selection
        subset = list(
            handle.read_segments(
                page.dataoffsets,
                page.databytecounts,
                indices=(0, 3),
                length=2,
            )
        )
        assert [index for _blob, index in subset] == [0, 3]

        # decoding the raw bytes through TiffPage.decode reproduces pixels
        for blob, index in segments:
            decoded, position, shape = page.decode(blob, index)
            _, _, h, w, _ = position
            edge_h = min(shape[1], data.shape[0] - h)
            edge_w = min(shape[2], data.shape[1] - w)
            assert array_equal(
                decoded[0, :edge_h, :edge_w, 0],
                data[h : h + edge_h, w : w + edge_w],
            )


def test_read_segments_short_read(tmp_file: Any) -> None:
    """Raw segment read silently returns the available short bytes."""
    data = pattern((20, 18), numpy.uint8)
    source = tmp_file('short_read_source.tif')
    imwrite(source, data, tile=(16, 16))
    with TiffFile(source) as tif:
        page = tif.pages.first
        last_offset = page.dataoffsets[-1]
        expected_count = page.databytecounts[-1]
    raw = open(source, 'rb').read()
    filename = tmp_file('short_read.tif')
    with open(filename, 'wb') as fh:
        fh.write(raw[: last_offset + 6])

    with TiffFile(filename) as tif:
        handle = tif.filehandle
        page = tif.pages.first
        segments = list(
            handle.read_segments(page.dataoffsets, page.databytecounts)
        )
        blob, index = segments[-1]
        assert index == 3
        assert blob is not None
        assert len(blob) == 6 < expected_count
        # TiffPage.decode turns the short segment into fillvalue padding
        decoded, _position, shape = page.decode(blob, index, _fillvalue=0)
        assert decoded.size == numpy.prod(shape)
        # the surviving bytes are the first samples of the decoded tile
        assert array_equal(
            decoded.flatten()[:6], numpy.frombuffer(blob, dtype=numpy.uint8)
        )
        assert (decoded.flatten()[6:] == 0).all()


###############################################################################
# Stream lifecycle: caller-owned file objects vs path-opened files
###############################################################################


def test_repeated_asarray_stable_on_stream(tmp_file: Any) -> None:
    """Repeated full and segmented decodes on one stream are stable."""
    data = pattern((20, 18), numpy.uint16)
    filename = tmp_file('repeat_stream.tif')
    imwrite(filename, data, tile=(16, 16), compression='adobe_deflate')
    raw = open(filename, 'rb').read()

    stream = io.BytesIO(raw)
    with TiffFile(stream) as tif:
        first = tif.asarray()
        for _ in range(3):
            assert array_equal(tif.asarray(), first)
            assert array_equal(tif.pages.first.asarray(), first)
            assert array_equal(assemble_segments(tif.pages.first), first)
        # interleaved page access does not corrupt a shared stream cursor
        segments = list(tif.pages.first.segments())
        assert array_equal(tif.asarray(), first)
        assert len(segments) == 4
    # TiffFile must not close a stream it did not open
    assert not stream.closed


def test_close_on_caller_stream_keeps_stream_usable() -> None:
    """Closing TiffFile on a passed stream does not close the stream."""
    data = pattern((6, 4), numpy.uint8)
    buffer = io.BytesIO()
    with TiffWriter(buffer) as writer:
        writer.write(data)
    raw = buffer.getvalue()

    stream = io.BytesIO(raw)
    tif = TiffFile(stream)
    assert array_equal(tif.asarray(), data)
    tif.close()
    # caller-owned stream remains open and seekable; reads still work
    assert not stream.closed
    assert stream.seek(0) == 0
    assert stream.read() == raw
    # TiffFile keeps serving data from the still-open stream without warning
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always')
        assert array_equal(tif.asarray(), data)
    assert records == []


def test_path_opened_file_reopens_with_warning(tmp_file: Any) -> None:
    """Path-based TiffFile reopens its own handle after close, warning once.

    The reopened handle is closed again once the read finishes, so the
    observable state after the call is closed again while the data is
    still returned correctly.

    """
    data = pattern((6, 4), numpy.uint8)
    filename = tmp_file('path_reopen.tif')
    imwrite(filename, data, rowsperstrip=2)

    tif = TiffFile(filename)
    assert array_equal(tif.asarray(), data)
    tif.close()
    assert tif.filehandle.closed

    # TiffPage.asarray temporarily reopens its own path-based handle
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always')
        image = tif.pages.first.asarray()
    assert array_equal(image, data)
    assert len(records) == 1
    assert issubclass(records[0].category, UserWarning)
    assert 'closed file' in str(records[0].message)
    # the temporary handle is closed again after reading
    assert tif.filehandle.closed

    # every closed-handle read warns again and still returns correct data
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always')
        assert array_equal(tif.pages.first.asarray(), data)
    assert len(records) == 1
    assert issubclass(records[0].category, UserWarning)

    # context manager closes path-opened handles as well
    with TiffFile(filename) as managed:
        assert array_equal(managed.asarray(), data)
    assert managed.filehandle.closed
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always')
        assert array_equal(managed.pages.first.asarray(), data)
    assert len(records) == 1
    assert issubclass(records[0].category, UserWarning)
    assert managed.filehandle.closed


def test_reopen_same_stream_content_repeatedly(tmp_file: Any) -> None:
    """Opening fresh streams over identical bytes gives identical pixels."""
    data = pattern((10, 6), numpy.uint16)
    filename = tmp_file('reopen_stream.tif')
    imwrite(filename, data, tile=(16, 16))
    raw = open(filename, 'rb').read()

    baseline = imread(filename)
    for _ in range(3):
        with TiffFile(io.BytesIO(raw)) as tif:
            assert array_equal(tif.asarray(), baseline)
            assert array_equal(assemble_segments(tif.pages.first), baseline)


def test_page_iteration_does_not_share_cursor_state(tmp_file: Any) -> None:
    """Alternating page reads on one TiffFile return each page's own data."""
    page_data = [
        pattern((2, 3), numpy.uint8),
        pattern((2, 3), numpy.uint8) + 10,
        pattern((2, 3), numpy.uint8) + 20,
    ]
    filename = tmp_file('page_cursor.tif')
    with TiffWriter(filename) as writer:
        for array in page_data:
            writer.write(array)

    with TiffFile(filename) as tif:
        assert len(tif.pages) == 3
        first = tif.pages[0].asarray()
        second = tif.pages[1].asarray()
        # re-reading page 0 after page 1 must not return page 1's bytes
        again = tif.pages[0].asarray()
        assert array_equal(first, again)
        assert not array_equal(first, second)
        for index, expected in enumerate(page_data):
            assert array_equal(tif.pages[index].asarray(), expected)
        # iterator and indexed access agree after mixed access
        for index, page in enumerate(tif.pages):
            assert array_equal(page.asarray(), page_data[index])


###############################################################################
# End-to-end entry points agree
###############################################################################


@pytest.mark.parametrize('byteorder', BYTEORDERS)
@pytest.mark.parametrize('compression', _compression_params())
def test_entry_points_agree(
    byteorder: str, compression: int | None, tmp_file: Any
) -> None:
    """imread, TiffFile.asarray, page.asarray and segments all agree."""
    tiled = pattern((18, 13), numpy.uint16)
    tile_name = tmp_file(f'entry_tiles_{byteorder}_{compression}.tif')
    kwargs = {} if compression is None else {'compression': compression}
    imwrite(tile_name, tiled, tile=(16, 16), byteorder=byteorder, **kwargs)
    assert array_equal(imread(tile_name), tiled)
    with TiffFile(tile_name) as tif:
        assert array_equal(tif.asarray(), tiled)
        page = tif.pages.first
        assert array_equal(page.asarray(), tiled)
        assert array_equal(assemble_segments(page), tiled)

    stripped = pattern((7, 5), numpy.uint16)
    strip_name = tmp_file(f'entry_strips_{byteorder}_{compression}.tif')
    imwrite(
        strip_name,
        stripped,
        rowsperstrip=3,
        byteorder=byteorder,
        **kwargs,
    )
    assert array_equal(imread(strip_name), stripped)
    with TiffFile(strip_name) as tif:
        assert array_equal(tif.asarray(), stripped)
        page = tif.pages.first
        assert array_equal(page.asarray(), stripped)
        assert array_equal(assemble_segments(page), stripped)
