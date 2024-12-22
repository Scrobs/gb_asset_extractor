#!/usr/bin/env python3
"""
Enhanced Game Boy ROM asset extractor supporting:
- High-quality graphics extraction with proper palettes
- Audio data extraction (Wave patterns and Sound effects)
- Sprite composition and animation frame extraction
- Background map reconstruction
"""

import os
import sys
import struct
import wave
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import logging
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class GBHeader:
    """Game Boy ROM header information."""
    title: str
    manufacturer_code: str
    cgb_flag: int
    new_licensee_code: str
    sgb_flag: int
    cartridge_type: int
    rom_size: int
    ram_size: int
    destination_code: int
    old_licensee_code: int
    mask_rom_version: int
    header_checksum: int
    global_checksum: int

class ROMReader:
    """Handles reading data from Game Boy ROM files."""

    HEADER_START = 0x100
    HEADER_SIZE = 0x50

    def __init__(self, rom_path: Path):
        self.rom_path = rom_path
        with open(rom_path, 'rb') as f:
            self.rom_data = f.read()
        self.header = self._parse_header()

    def _parse_header(self) -> GBHeader:
        """Parse the ROM header information."""
        header_data = self.rom_data[self.HEADER_START:self.HEADER_START + self.HEADER_SIZE]
        return GBHeader(
            title=header_data[0x34:0x43].decode('ascii').rstrip('\x00'),
            manufacturer_code=header_data[0x3F:0x43].decode('ascii'),
            cgb_flag=header_data[0x43],
            new_licensee_code=header_data[0x44:0x46].decode('ascii'),
            sgb_flag=header_data[0x46],
            cartridge_type=header_data[0x47],
            rom_size=header_data[0x48],
            ram_size=header_data[0x49],
            destination_code=header_data[0x4A],
            old_licensee_code=header_data[0x4B],
            mask_rom_version=header_data[0x4C],
            header_checksum=header_data[0x4D],
            global_checksum=struct.unpack('>H', header_data[0x4E:0x50])[0]
        )

class SoundEngine:
    """Handles Game Boy sound chip emulation and extraction."""

    SAMPLE_RATE = 44100
    FRAME_RATE = 60

    def __init__(self):
        self.samples_per_frame = self.SAMPLE_RATE // self.FRAME_RATE

    def generate_square_wave(self, frequency: float, duty_cycle: float = 0.5) -> np.ndarray:
        """Generate square wave samples."""
        t = np.linspace(0, 1, self.samples_per_frame, endpoint=False)
        return np.where(t % (1/frequency) < duty_cycle/frequency, 1.0, -1.0)

    def apply_envelope(self, samples: np.ndarray, initial_volume: float,
                      direction: int, sweep_pace: int) -> np.ndarray:
        """Apply volume envelope to samples."""
        if sweep_pace == 0:
            return samples * initial_volume

        steps = len(samples)
        envelope = np.linspace(initial_volume,
                             initial_volume + direction * sweep_pace/15,
                             steps)
        envelope = np.clip(envelope, 0, 1)
        return samples * envelope

class GraphicsExtractor:
    """Handles extraction of graphics assets."""

    TILE_SIZE = 8
    BYTES_PER_TILE = 16
    TILES_PER_BANK = 384

    def __init__(self, rom_reader: ROMReader):
        self.rom_reader = rom_reader
        self.default_palettes = {
            'original': [
                (255, 255, 255),  # White
                (192, 192, 192),  # Light gray
                (96, 96, 96),     # Dark gray
                (0, 0, 0)         # Black
            ],
            'green': [
                (155, 188, 15),   # Light green
                (139, 172, 15),   # Green
                (48, 98, 48),     # Dark green
                (15, 56, 15)      # Darkest green
            ],
            'pocket': [
                (155, 188, 15),   # Greenish
                (139, 172, 15),   # Light green
                (48, 98, 48),     # Dark green
                (15, 56, 15)      # Darkest green
            ],
            'sgb': [  # Super Game Boy enhanced colors
                (255, 255, 255),  # White
                (255, 192, 192),  # Light red
                (192, 128, 128),  # Dark red
                (0, 0, 0)         # Black
            ]
        }

    def extract_tile(self, offset: int) -> np.ndarray:
        """Extract a single 8x8 tile from ROM data."""
        tile = np.zeros((8, 8), dtype=np.uint8)
        for y in range(8):
            byte1 = self.rom_reader.rom_data[offset + y * 2]
            byte2 = self.rom_reader.rom_data[offset + y * 2 + 1]
            for x in range(8):
                bit1 = (byte1 >> (7 - x)) & 1
                bit2 = (byte2 >> (7 - x)) & 1
                tile[y, x] = bit1 | (bit2 << 1)
        return tile

    def extract_sprite(self, offset: int, size: int = 8) -> np.ndarray:
        """Extract a sprite of arbitrary size."""
        if size not in [8, 16]:
            raise ValueError("Sprite size must be 8 or 16 pixels")

        sprite = np.zeros((size, 8), dtype=np.uint8)
        num_tiles = size // 8

        for tile_idx in range(num_tiles):
            tile_offset = offset + tile_idx * self.BYTES_PER_TILE
            tile = self.extract_tile(tile_offset)
            sprite[tile_idx*8:(tile_idx+1)*8, :] = tile

        return sprite

    def extract_tilemap(self, offset: int, width: int, height: int) -> np.ndarray:
        """Extract and compose a tilemap."""
        tilemap = np.zeros((height * 8, width * 8), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                tile_idx = self.rom_reader.rom_data[offset + y * width + x]
                tile_offset = 0x8000 + tile_idx * self.BYTES_PER_TILE

                if tile_offset + self.BYTES_PER_TILE <= len(self.rom_reader.rom_data):
                    tile = self.extract_tile(tile_offset)
                    tilemap[y*8:(y+1)*8, x*8:(x+1)*8] = tile

        return tilemap

    def extract_tileset(self, bank_offset: int, num_tiles: int,
                       palette_name: str = 'original') -> Image.Image:
        """Extract and compose a complete tileset."""
        tiles_per_row = 16
        rows = (num_tiles + tiles_per_row - 1) // tiles_per_row

        width = tiles_per_row * self.TILE_SIZE
        height = rows * self.TILE_SIZE
        tileset = np.zeros((height, width), dtype=np.uint8)

        for tile_idx in range(num_tiles):
            tile_offset = bank_offset + tile_idx * self.BYTES_PER_TILE
            if tile_offset + self.BYTES_PER_TILE > len(self.rom_reader.rom_data):
                break

            tile = self.extract_tile(tile_offset)
            row = tile_idx // tiles_per_row
            col = tile_idx % tiles_per_row

            y_start = row * self.TILE_SIZE
            x_start = col * self.TILE_SIZE
            tileset[y_start:y_start + self.TILE_SIZE,
                   x_start:x_start + self.TILE_SIZE] = tile

        img = Image.fromarray(tileset, mode='P')
        img.putpalette(sum(self.default_palettes[palette_name], ()))
        return img

    def detect_sprite_table(self) -> Optional[int]:
        """Attempt to detect the sprite attribute table location."""
        # Common sprite table signatures
        signatures = [
            (0xC000, 0x100),  # Common location in many games
            (0xFE00, 0x100)   # Hardware OAM location
        ]

        for offset, size in signatures:
            if offset + size <= len(self.rom_reader.rom_data):
                # Check for valid sprite data patterns
                valid_entries = 0
                for i in range(0, size, 4):
                    y_pos = self.rom_reader.rom_data[offset + i]
                    x_pos = self.rom_reader.rom_data[offset + i + 1]
                    tile_idx = self.rom_reader.rom_data[offset + i + 2]
                    attrs = self.rom_reader.rom_data[offset + i + 3]

                    # Basic validation of sprite attributes
                    if (16 <= y_pos <= 160 and
                        8 <= x_pos <= 168 and
                        attrs & 0xF0 == 0):  # Upper bits should be 0
                        valid_entries += 1

                if valid_entries >= 8:  # Arbitrary threshold
                    return offset

        return None

class AudioExtractor:
    """Handles extraction of audio assets."""

    WAVE_PATTERN_LENGTH = 16
    WAVE_RAM_SIZE = 16

    def __init__(self, rom_reader: ROMReader):
        self.rom_reader = rom_reader
        self.sound_engine = SoundEngine()

    def extract_wave_pattern(self, offset: int, length: int = WAVE_PATTERN_LENGTH) -> bytes:
        """Extract wave pattern data."""
        return self.rom_reader.rom_data[offset:offset + length]

    def detect_sound_data(self) -> List[Tuple[int, int]]:
        """Attempt to detect sound data locations."""
        sound_locations = []

        # Look for common sound engine signatures
        signatures = [
            bytes([0x11, 0x83, 0x00]),  # Common frequency value
            bytes([0x80, 0xFF, 0x0F]),  # Wave RAM initialization
        ]

        for i in range(0, len(self.rom_reader.rom_data) - 16):
            for sig in signatures:
                if self.rom_reader.rom_data[i:i+len(sig)] == sig:
                    # Validate surrounding data
                    if all(0 <= x <= 0xFF for x in self.rom_reader.rom_data[i:i+16]):
                        sound_locations.append((i, 16))

        return sound_locations

    def save_wave_pattern(self, pattern: bytes, output_path: Path) -> None:
        """Save wave pattern as WAV file."""
        samples = np.array([(x >> 4) / 7.5 - 1 for x in pattern], dtype=np.float32)
        samples = np.repeat(samples, self.sound_engine.samples_per_frame // len(samples))

        # Apply basic envelope
        samples = self.sound_engine.apply_envelope(samples, 1.0, -1, 2)

        # Convert to 16-bit PCM
        samples = (samples * 32767).astype(np.int16)

        with wave.open(str(output_path), 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sound_engine.SAMPLE_RATE)
            wav_file.writeframes(samples.tobytes())

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} ROM_FILE OUTPUT_DIR")
        sys.exit(1)

    rom_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    try:
        # Initialize extractors
        rom_reader = ROMReader(rom_path)
        graphics_extractor = GraphicsExtractor(rom_reader)
        audio_extractor = AudioExtractor(rom_reader)

        # Create output directories
        graphics_dir = output_dir / 'graphics'
        audio_dir = output_dir / 'audio'
        sprite_dir = output_dir / 'sprites'
        tilemap_dir = output_dir / 'tilemaps'
        for dir_path in [graphics_dir, audio_dir, sprite_dir, tilemap_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Extract ROM header information
        header_info = os.path.join(output_dir, 'rom_info.txt')
        with open(header_info, 'w') as f:
            for field, value in rom_reader.header.__dict__.items():
                f.write(f"{field}: {value}\n")
        logger.info(f"Saved ROM header information to {header_info}")

        # Extract graphics
        logger.info("Extracting graphics...")
        bank_offsets = [0x4000, 0x8000, 0xC000]  # Common tile bank locations
        for bank_idx, offset in enumerate(bank_offsets):
            try:
                # Extract with multiple palettes
                for palette_name in ['original', 'green', 'pocket', 'sgb']:
                    tileset = graphics_extractor.extract_tileset(
                        offset,
                        graphics_extractor.TILES_PER_BANK,
                        palette_name
                    )
                    output_path = graphics_dir / f'tileset_bank_{bank_idx}_{palette_name}.png'
                    tileset.save(output_path)
                    logger.info(f"Saved tileset from bank {bank_idx} with {palette_name} palette")
            except Exception as e:
                logger.error(f"Error extracting bank {bank_idx}: {e}")

        # Extract sprites
        logger.info("Extracting sprites...")
        sprite_table_offset = graphics_extractor.detect_sprite_table()
        if sprite_table_offset:
            logger.info(f"Found sprite table at 0x{sprite_table_offset:04X}")
            for sprite_idx in range(40):  # Hardware supports 40 sprites
                try:
                    sprite_8x8 = graphics_extractor.extract_sprite(sprite_table_offset + sprite_idx * 4, 8)
                    sprite_8x16 = graphics_extractor.extract_sprite(sprite_table_offset + sprite_idx * 4, 16)

                    for palette_name in ['original', 'green']:
                        # Save 8x8 sprite
                        img_8x8 = Image.fromarray(sprite_8x8, mode='P')
                        img_8x8.putpalette(sum(graphics_extractor.default_palettes[palette_name], ()))
                        img_8x8.save(sprite_dir / f'sprite_{sprite_idx:02d}_8x8_{palette_name}.png')

                        # Save 8x16 sprite
                        img_8x16 = Image.fromarray(sprite_8x16, mode='P')
                        img_8x16.putpalette(sum(graphics_extractor.default_palettes[palette_name], ()))
                        img_8x16.save(sprite_dir / f'sprite_{sprite_idx:02d}_8x16_{palette_name}.png')
                except Exception as e:
                    logger.error(f"Error extracting sprite {sprite_idx}: {e}")

        # Extract tilemaps
        logger.info("Extracting tilemaps...")
        tilemap_locations = [(0x9800, 32, 32), (0x9C00, 32, 32)]  # Standard GB tilemap locations
        for idx, (offset, width, height) in enumerate(tilemap_locations):
            try:
                tilemap = graphics_extractor.extract_tilemap(offset, width, height)
                for palette_name in ['original', 'green']:
                    img = Image.fromarray(tilemap, mode='P')
                    img.putpalette(sum(graphics_extractor.default_palettes[palette_name], ()))
                    img.save(tilemap_dir / f'tilemap_{idx}_{palette_name}.png')
                logger.info(f"Saved tilemap {idx}")
            except Exception as e:
                logger.error(f"Error extracting tilemap {idx}: {e}")

        # Extract audio patterns
        logger.info("Extracting audio patterns...")
        # First, try to detect sound data locations
        sound_locations = audio_extractor.detect_sound_data()
        if sound_locations:
            logger.info(f"Found {len(sound_locations)} potential sound patterns")
            for idx, (offset, length) in enumerate(sound_locations):
                try:
                    pattern = audio_extractor.extract_wave_pattern(offset, length)
                    output_path = audio_dir / f'sound_pattern_{idx:02d}.wav'
                    audio_extractor.save_wave_pattern(pattern, output_path)
                    logger.info(f"Saved sound pattern {idx} to {output_path}")
                except Exception as e:
                    logger.error(f"Error extracting sound pattern {idx}: {e}")
        else:
            logger.warning("No sound patterns detected, trying fixed locations...")
            # Try fixed locations as fallback
            audio_locations = [
                (0x4000, 'wave_bank_0'),
                (0x8000, 'wave_bank_1')
            ]
            for offset, name in audio_locations:
                try:
                    pattern = audio_extractor.extract_wave_pattern(offset)
                    output_path = audio_dir / f'{name}.wav'
                    audio_extractor.save_wave_pattern(pattern, output_path)
                    logger.info(f"Saved audio pattern to {output_path}")
                except Exception as e:
                    logger.error(f"Error extracting audio pattern {name}: {e}")

        logger.info("Asset extraction complete!")

    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
