# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains Python examples for the PLUX biosignals acquisition API. It provides sample code for collecting real-time biosignal data from PLUX devices using platform-specific native libraries.

## Core Architecture

### PLUX API Integration
- The repository uses precompiled native libraries located in `PLUX-API-Python3/` with platform-specific subdirectories
- Platform detection logic automatically selects the correct library path based on OS and Python version
- The `plux` module provides device communication interfaces through `SignalsDev` and `MemoryDev` base classes

### Device Classes
All examples implement custom device classes that inherit from PLUX base classes:
- `plux.SignalsDev` - For real-time signal acquisition
- `plux.MemoryDev` - For scheduled recordings and data download

The core pattern involves:
1. Platform-specific library path setup via `sys.path.append()`
2. Custom device class with `onRawFrame()` callback for data processing
3. Device lifecycle: `start()` → `loop()` → `stop()` → `close()`

### Channel Configuration
- Analog channels (ports 1-8): Use channel mask codes (0x01 for 1 channel, 0x03 for 2, etc.)
- Digital channels (port 9): For SpO2/fNIRS sensors with dual derivations (RED/INFRARED)
- `plux.Source` objects define channel properties (port, resolution, frequency divisor, channel mask)

## Running Examples

Each example can be run directly with Python and accepts command-line arguments:

```bash
# Basic single device acquisition
python OneDeviceAcquisitionExample.py [address] [duration] [frequency] [channel_code]

# Special channels (digital sensors)
python OneDeviceSpecialChannelsExample.py [address] [duration] [frequency]

# Multi-device threading
python MultipleDeviceThreadingExample.py

# Scheduled acquisition
python ScheduleAcquisitionExample.py [address] [start_delay] [duration] [frequency]

# Download recorded data
python DownloadAcquisitionExample.py [address]

# BITalino specific acquisition
python OneBITalinoAcquisitionExample.py [address] [duration] [frequency] [active_ports]
```

Default device address: `BTH00:07:80:4D:2E:76`

## Platform Compatibility

The repository supports:
- **Windows**: Win32/Win64 with Python 3.7-3.12
- **macOS**: Intel and M1 architectures with Python 3.7-3.12
- **Linux**: x64 and ARM32/ARM64 with Python 3.8-3.11

Special handling for macOS Monterey (12.x) requires Python ≥ 3.10.

## Key Technical Details

### Frequency Limitations
Channel count affects maximum sampling frequency:
- 1-2 channels: up to 8000Hz/5000Hz
- 3-4 channels: up to 4000Hz/3000Hz  
- 5-8 channels: up to 3000Hz/2000Hz

### Threading on macOS
Multi-device examples require special macOS handling with `plux.MacOS.runMainLoop()` and `plux.MacOS.stopMainLoop()`.

### Data Processing
The `onRawFrame(nSeq, data)` callback receives sequence numbers and raw data arrays. Examples typically print every 2000th sample to avoid overwhelming output.