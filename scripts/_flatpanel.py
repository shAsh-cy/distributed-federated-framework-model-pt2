"""A dependency-free PNG writer and a 5x7 bitmap font.

Used to draw the *visibly placeholder* panels that stand in for the
gradient-inversion figures until the real ones land (see
docs/inversion/README.md). Nothing here is a plotting library and nothing
here should grow into one: it exists so that a placeholder can never be
mistaken for a reconstruction, and so that generating one needs no
third-party dependency at all.

Pure stdlib: zlib and struct.
"""

from __future__ import annotations

import struct
import zlib

RGB = tuple[int, int, int]

# The instrument palette, duplicated here because these files are generated
# outside the dashboard build. Keep in step with dashboard/src/styles/tokens.css.
GROUND_RAISED: RGB = (0xF0, 0xF2, 0xEE)
INK: RGB = (0x3A, 0x3D, 0x3C)
RULE: RGB = (0xC6, 0xCB, 0xC2)

# 5x7 uppercase font. Only the glyphs the placeholders need; a missing glyph
# raises rather than rendering blank, so a typo in a caption is a loud failure.
_FONT: dict[str, tuple[str, ...]] = {
    " ": ("     ", "     ", "     ", "     ", "     ", "     ", "     "),
    "-": ("     ", "     ", "     ", "#####", "     ", "     ", "     "),
    ".": ("     ", "     ", "     ", "     ", "     ", "  ## ", "  ## "),
    "=": ("     ", "     ", "#####", "     ", "#####", "     ", "     "),
    "/": ("    #", "    #", "   # ", "  #  ", " #   ", "#    ", "#    "),
    "0": (" ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### "),
    "1": ("  #  ", " ##  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "),
    "2": (" ### ", "#   #", "    #", "   # ", "  #  ", " #   ", "#####"),
    "3": ("#####", "   # ", "  #  ", "   # ", "    #", "#   #", " ### "),
    "4": ("   # ", "  ## ", " # # ", "#  # ", "#####", "   # ", "   # "),
    "5": ("#####", "#    ", "#### ", "    #", "    #", "#   #", " ### "),
    "6": ("  ## ", " #   ", "#    ", "#### ", "#   #", "#   #", " ### "),
    "7": ("#####", "    #", "   # ", "  #  ", " #   ", " #   ", " #   "),
    "8": (" ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### "),
    "9": (" ### ", "#   #", "#   #", " ####", "    #", "   # ", " ##  "),
    "A": (" ### ", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"),
    "B": ("#### ", "#   #", "#   #", "#### ", "#   #", "#   #", "#### "),
    "C": (" ### ", "#   #", "#    ", "#    ", "#    ", "#   #", " ### "),
    "D": ("###  ", "#  # ", "#   #", "#   #", "#   #", "#  # ", "###  "),
    "E": ("#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#####"),
    "F": ("#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#    "),
    "G": (" ### ", "#   #", "#    ", "#  ##", "#   #", "#   #", " ####"),
    "H": ("#   #", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"),
    "I": (" ### ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "),
    "J": ("  ###", "   # ", "   # ", "   # ", "   # ", "#  # ", " ##  "),
    "K": ("#   #", "#  # ", "# #  ", "##   ", "# #  ", "#  # ", "#   #"),
    "L": ("#    ", "#    ", "#    ", "#    ", "#    ", "#    ", "#####"),
    "M": ("#   #", "## ##", "# # #", "# # #", "#   #", "#   #", "#   #"),
    "N": ("#   #", "##  #", "# # #", "# # #", "#  ##", "#   #", "#   #"),
    "O": (" ### ", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "),
    "P": ("#### ", "#   #", "#   #", "#### ", "#    ", "#    ", "#    "),
    "Q": (" ### ", "#   #", "#   #", "#   #", "# # #", "#  # ", " ## #"),
    "R": ("#### ", "#   #", "#   #", "#### ", "# #  ", "#  # ", "#   #"),
    "S": (" ####", "#    ", "#    ", " ### ", "    #", "    #", "#### "),
    "T": ("#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  "),
    "U": ("#   #", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "),
    "V": ("#   #", "#   #", "#   #", "#   #", "#   #", " # # ", "  #  "),
    "W": ("#   #", "#   #", "#   #", "# # #", "# # #", "## ##", "#   #"),
    "X": ("#   #", "#   #", " # # ", "  #  ", " # # ", "#   #", "#   #"),
    "Y": ("#   #", "#   #", " # # ", "  #  ", "  #  ", "  #  ", "  #  "),
    "Z": ("#####", "    #", "   # ", "  #  ", " #   ", "#    ", "#####"),
}

GLYPH_W = 5
GLYPH_H = 7


class Canvas:
    """A flat RGB raster with rectangle and bitmap-text drawing."""

    def __init__(self, width: int, height: int, background: RGB) -> None:
        self.width = width
        self.height = height
        self.pixels = bytearray(bytes(background) * (width * height))

    def rect(self, x: int, y: int, w: int, h: int, colour: RGB) -> None:
        # Clamp to the canvas: an unclamped row slice would wrap onto the next
        # scanline, which is how a too-wide caption silently corrupts a panel.
        x0, x1 = max(0, x), min(self.width, x + w)
        if x1 <= x0:
            return
        row = bytes(colour) * (x1 - x0)
        for yy in range(max(0, y), min(y + h, self.height)):
            start = (yy * self.width + x0) * 3
            self.pixels[start : start + len(row)] = row

    def frame(self, inset: int, thickness: int, colour: RGB) -> None:
        w, h = self.width - 2 * inset, self.height - 2 * inset
        self.rect(inset, inset, w, thickness, colour)
        self.rect(inset, inset + h - thickness, w, thickness, colour)
        self.rect(inset, inset, thickness, h, colour)
        self.rect(inset + w - thickness, inset, thickness, h, colour)

    def text(self, x: int, y: int, message: str, scale: int, colour: RGB) -> int:
        """Draw `message` with its top-left at (x, y). Returns the width drawn."""
        cursor = x
        for char in message.upper():
            glyph = _FONT.get(char)
            if glyph is None:
                raise KeyError(f"no 5x7 glyph for {char!r}; add one to _FONT")
            for row, bits in enumerate(glyph):
                for col, bit in enumerate(bits):
                    if bit == "#":
                        self.rect(cursor + col * scale, y + row * scale, scale, scale, colour)
            cursor += (GLYPH_W + 1) * scale
        return cursor - x

    def text_width(self, message: str, scale: int) -> int:
        return max(0, len(message) * (GLYPH_W + 1) * scale - scale)

    def text_centred(self, y: int, message: str, scale: int, colour: RGB) -> None:
        width = self.text_width(message, scale)
        if width > self.width:
            raise ValueError(f"{message!r} at scale {scale} is {width}px, wider than the panel")
        self.text((self.width - width) // 2, y, message, scale, colour)

    def to_png(self) -> bytes:
        raw = bytearray()
        stride = self.width * 3
        for y in range(self.height):
            raw.append(0)  # filter type 0 (None) — these images are tiny and flat
            raw.extend(self.pixels[y * stride : (y + 1) * stride])
        return _png_bytes(self.width, self.height, bytes(raw))


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


def _png_bytes(width: int, height: int, raw: bytes) -> bytes:
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    # Fixed compression level so regenerating produces byte-identical output.
    body = zlib.compress(raw, 9)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", body)
        + _chunk(b"IEND", b"")
    )
