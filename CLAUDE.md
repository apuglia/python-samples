# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python API sample repository for PLUX biosignal acquisition devices (biosignalsplux, BITalino). It contains example scripts demonstrating how to interface with PLUX devices for real-time data acquisition, scheduled recording, and data download.

## Core Architecture

The codebase follows a consistent pattern across all examples:

### Platform-Specific Library Loading
All examples use dynamic library loading based on the platform and Python version:
- **Windows**: `Win32_37/`, `Win64_37/`, `Win64_38/`, `Win64_39/`, `Win64_310/`
- **macOS**: `MacOS/Intel37/`, `MacOS/Intel38/`, `MacOS/Intel39/`, `MacOS/Intel310/`, `M1_37/`, `M1_39/`, `M1_310/`, `M1_311/`, `M1_312/`
- **Linux**: `Linux64/`, `LinuxARM32/`, `LinuxARM32_311/`, `LinuxARM64_38/`, `LinuxARM64_39/`

The library path is automatically determined using the `osDic` dictionary and appended to `sys.path`.

### Device Classes
Two main device classes are used:
- **`plux.SignalsDev`**: For real-time signal acquisition
- **`plux.MemoryDev`**: For scheduled recording and data download

### Channel Configuration
- **Channel codes**: Hexadecimal values (0x01 for 1 channel, 0x03 for 2 channels, etc.)
- **Active ports**: List of port numbers [1, 2, 3, 4, 5, 6] for BITalino
- **Source objects**: For advanced channel configuration with `plux.Source()`

### Data Handling
All examples implement `onRawFrame(self, nSeq, data)` callback for processing incoming data frames.

## Running Examples

### Basic Usage
```bash
python OneDeviceAcquisitionExample.py [address] [duration] [frequency] [channel_code]
python OneDeviceSpecialChannelsExample.py [address] [duration] [frequency]
python OneBITalinoAcquisitionExample.py [address] [duration] [frequency] [active_ports]
```

### Advanced Examples
```bash
python ScheduleAcquisitionExample.py [address] [start_in_seconds] [duration] [frequency]
python DownloadAcquisitionExample.py [address]
```

### Multi-Device Threading
The `MultipleDeviceThreadingExample.py` is configured to run with hardcoded device addresses and parameters.

## Device Addresses

Examples use Bluetooth addresses in the format: `BTH00:07:80:XX:XX:XX`

## Channel Specifications

### Analog Channels (Ports 1-8)
- **Channel mask**: `0x01` (single derivation)
- **Resolution**: 16-bit ADC
- **Frequency divisor**: 1 (no subsampling)

### Digital Channels (Port 9)
- **Channel mask**: `0x03` (dual derivation for SpO2/fNIRS)
- **Resolution**: 16-bit ADC
- **Produces**: RED and INFRARED signals

### Frequency Limits
- 1 channel: 8000 Hz max
- 2 channels: 5000 Hz max
- 3 channels: 4000 Hz max
- 4 channels: 3000 Hz max
- 5-6 channels: 2000-3000 Hz max
- 7-8 channels: 2000 Hz max

## macOS Considerations

For macOS systems, especially Monterey (version 12.x):
- Minimum Python version: 3.10
- Requires `plux.MacOS.runMainLoop()` and `plux.MacOS.stopMainLoop()` for threading
- Uses `bth_macprocess` helper for Bluetooth communication

## Development Notes

- No package management files (package.json, requirements.txt) - dependencies are handled via compiled libraries
- All examples are self-contained and can be run directly
- The `plux` module is imported from platform-specific compiled libraries
- Threading examples require special handling on macOS for proper Bluetooth communication