# eye-track-rs

An ultra-lightweight, real-time computer vision engine built entirely in **pure Rust** for robust eye tracking without the overhead of heavy ML frameworks. Achieves sub-millisecond processing with **27MB RAM**, **~30 FPS**, and **18-27% CPU utilization**.

## Overview

`eye-track-rs` is a **heuristic-based pupil tracking system** designed for edge computing and resource-constrained environments. Rather than relying on neural networks or complex computer vision pipelines, it uses:

- **Spatial funnel calibration**: Rapid, zero-learning initialization that finds pupils in ~3 seconds
- **Adaptive template matching**: SAD (Sum of Absolute Differences) algorithm for frame-to-frame tracking
- **Immutable anchor anti-drift**: Prevents template corruption from eyebrow/hair tracking without interrupting inference
- **Mathematical blink rejection**: Auto-recovery when eyes close (tracks up to 1.5 seconds of occlusion)
- **Online learning**: Gradual template adaptation at 0.5% per frame to handle lighting changes

The result: **high-fidelity pupil center detection** with minimal computational overhead.

## Performance Metrics

| Metric | Value |
|--------|-------|
| **Frame Rate** | ~30 FPS |
| **RAM Usage** | 27 MB |
| **CPU Usage** | 18-27% (single core) |
| **Latency** | <33ms per frame |
| **Template Size** | 40×40 pixels per eye |
| **Search Range** | 15-pixel radius per frame |

*Tested on consumer-grade hardware with 1280×720 camera input.*

---

## 🔒 Privacy & Security

### What This Code Does (and Doesn't Do)

**This application:**
- ✅ Processes video **locally on your machine** (no network calls)
- ✅ Does **not send data** to any server or cloud service
- ✅ Does **not store** frames, templates, or tracking data to disk
- ✅ Runs with **no external dependencies** for ML models or web APIs
- ✅ Is **open-source**—all code is visible and auditable

**This application does NOT:**
- ❌ Transmit biometric data
- ❌ Perform remote gaze estimation or prediction
- ❌ Store eye templates or calibration data
- ❌ Make network requests
- ❌ Access the filesystem (except process telemetry)
- ❌ Require internet connectivity

### Your Control

Because `eye-track-rs` is **pure Rust with no neural network weights or pretrained models**, there is:
- **No proprietary black box** that could secretly process data
- **No model updates** that could change behavior without your knowledge
- **No licensing verification** that phones home
- **No data collection mechanisms** built into the code

You can read every line of tracking logic. You can compile it yourself. You can run it offline. You can modify it.

### For Employers & Institutional Use

This library is suitable for:
- **Privacy-first eye-tracking UI** (accessibility features, gaze-based interaction)
- **On-device analytics** (fatigue detection, attention measurement) with no data exfiltration
- **Research environments** where data cannot leave the lab
- **Resource-constrained devices** (embedded systems, mobile edge)

**Example: Fair Use Scenarios**
- Detecting user fatigue in a driving assistance system (local processing only)
- Enabling gaze-based accessibility for users with motor disabilities
- Research on attention patterns without uploading data
- Development of low-latency eye-tracking UI for gaming

---

## Getting Started

### Prerequisites

- **Rust 1.70+** ([install here](https://rustup.rs/))
- A connected **webcam** (supports YUYV format)
- **Windows**, **Linux**, or **macOS** (platform support varies with camera drivers)

### Build & Run

```bash
git clone https://github.com/piresjoaov/eye-track-rs.git
cd eye-track-rs
cargo build --release
cargo run --release
```

### First Run

1. **Launch the application**
   - A window opens with your webcam feed
   - Green boxes show the calibration funnels (left and right eyes)

2. **Calibration Phase** (~3 seconds)
   - Look straight ahead into the camera
   - Stay still while the system finds your pupils
   - Once pupils are detected, red dots appear on-screen

3. **Tracking Phase**
   - Red dots follow your eye centers in real-time
   - Blink naturally—the system pauses during blinks
   - If you move your head too far, the system recalibrates automatically

4. **Exit**
   - Press `ESC` to close

### Real-Time Telemetry

The console prints live performance stats:

```
[Pure Telemetry] FPS: 30 | CPU: 22.3% | RAM: 27 MB
[Pure Telemetry] FPS: 30 | CPU: 19.8% | RAM: 27 MB
```

---

## How It Works

### 1. Calibration: The Spatial Funnel

During the first 90 frames (~3 seconds):
- The system scans predefined regions: left eye (60×30 px) and right eye (60×30 px)
- It finds the **darkest point** in each region (the pupil)
- Tightens tolerance to a 15-shade window to isolate the pupil
- Computes the **gravitational center** of dark pixels

```
High tolerance → Finds approximate pupil
    ↓
Tight tolerance (15 shades) → Isolates pupil core
    ↓
Pixel-weighted center → Subpixel accuracy
```

### 2. Tracking: Adaptive Template Matching

Every frame after calibration:
- **Template**: 40×40 grayscale patch centered on the last detected pupil
- **Search**: Shift the template ±15 pixels in X and Y
- **Metric**: SAD (Sum of Absolute Differences) between template and frame
- **Update**: Move position to the shift with minimum SAD

### 3. Anti-Drift: The Anchor Mechanism

One of the novel aspects of this tracker is the **immutable anchor**:

- **Original anchor**: The first template captured during calibration (the "lifesaver")
- **Active template**: Adapts frame-to-frame at 0.5% learning rate
- **Drift detection**: If the active template drifts >45 shades from the anchor, it silently resets
- **Benefit**: Prevents slow drift into eyebrows/hair without interrupting tracking

```
Frame 0: Pupil detected → Save anchor template
Frame 1-N: Track with adaptive template, compare to anchor
Frame 200: Template drifting to eyebrow? Reset to anchor, continue tracking
```

### 4. Blink Recovery

When eyes close or move too fast:
- SAD exceeds threshold → `frames_lost` counter increments
- If `frames_lost > 45` (~1.5 seconds), assume true occlusion → recalibrate
- During blink: position freezes, no new template updates

---

## Architecture

```
Input: Raw Camera Frame (YUYV format)
    ↓
[VideoStream] → Raw pixel buffer
    ↓
[YUYV to Grayscale] → Efficient format conversion
    ↓
[PureRustTracker]
    ├─ State: Calibrating | Tracking
    ├─ left_eye: EyeTracker { template, anchor, position }
    └─ right_eye: EyeTracker { template, anchor, position }
    ↓
[Stabilizer] → Low-pass filter (α=0.4)
    ↓
[Render] → Draw pupils on frame
    ↓
Output: Display window + Console telemetry
```

### Key Components

| Module | Role |
|--------|------|
| `main.rs` | Main loop, rendering, telemetry |
| `camera.rs` | Camera abstraction (nokhwa wrapper) |
| `EyeTracker` | Core tracking state machine |
| `PureRustTracker` | Dual-eye orchestration |
| `Stabilizer` | Exponential moving average filter |

---

## Dependencies

| Crate | Purpose | Size |
|-------|---------|------|
| **nokhwa** 0.10.0 | Camera capture | Low-level camera API |
| **minifb** 0.24.0 | Window rendering | Minimal framebuffer library |
| **sysinfo** 0.30 | Process monitoring | Telemetry only |

All dependencies are **zero-ML, zero-network, zero-data-collection**.

---

## Limitations & Future Work

### Current Limitations

- **Single calibration state**: Assumes fixed lighting and camera position
- **No head movement compensation**: Assumes face is relatively stationary
- **Pupil-only tracking**: Does not estimate gaze direction (3D gaze angle)
- **Template size fixed**: 40×40 pixels (optimized for typical webcam resolution)
- **Windows-only (MSMF)**: Linux/Mac require different camera backends
- **No persistent calibration**: Recalibration needed on app restart

### Planned Improvements

- [ ] Configuration file for threshold tuning
- [ ] Gaze angle estimation (requires eye-camera geometry calibration)
- [ ] Multi-resolution template matching for robustness
- [ ] Head pose compensation
- [ ] Calibration persistence (save/load anchor templates)
- [ ] Network output API (for external applications)
- [ ] Cross-platform camera support (v4l2 on Linux, AVFoundation on macOS)

---

## Use Cases

### ✅ Well-Suited For

1. **Accessibility**: Eye-gaze cursor control for motor-impaired users
2. **UI Research**: Attention tracking in UX studies (local processing)
3. **Fatigue Detection**: Driver drowsiness monitoring
4. **Gaming**: Gaze-based interactions
5. **Edge Devices**: Embedded systems, IoT, resource-constrained platforms
6. **Privacy-First Analytics**: On-device attention measurement without data exfiltration

### ⚠️ Not Suitable For

- High-precision gaze estimation without additional calibration
- Very low-light environments (<30 lux)
- Fast head movements (>5 degrees/frame)
- Extreme lighting changes mid-session
- Users wearing very dark sunglasses

---

## Building from Source

```bash
# Clone
git clone https://github.com/piresjoaov/eye-track-rs.git
cd eye-track-rs

# Debug build (faster compile, slower runtime)
cargo build
cargo run

# Release build (optimized, ~2x faster)
cargo build --release
cargo run --release

# Check code without building
cargo check

# Run tests (when available)
cargo test
```

### Platform-Specific Notes

**Windows**: Works out-of-the-box with MSMF (Media Foundation).

**Linux**: Requires `libv4l2-dev`:
```bash
sudo apt-get install libv4l2-dev
```

**macOS**: Requires AVFoundation (built-in).

---

## Performance Tuning

### If You Want Better Accuracy

Reduce search range and increase template size (in `main.rs`):

```rust
let search_range = 10_isize;  // Smaller = more precise, slower
let template_w = 50;          // Larger = more context, more computation
```

### If You Want Better Performance

Increase search range or reduce refresh rate:

```rust
let search_range = 20_isize;  // Larger = faster, less precise
window.limit_update_rate(Some(Duration::from_millis(50))); // 20 FPS
```

---

## Contributing

This is an early-stage project. If you find bugs or have ideas:

1. **Open an issue** with:
   - Your camera specs
   - OS and Rust version
   - What failed and how

2. **Submit a PR** with:
   - Tests for your change
   - Updated documentation
   - Performance impact analysis

---

## License

MIT License. See [LICENSE](LICENSE) file for details.

You are free to:
- ✅ Use in commercial projects
- ✅ Modify and redistribute
- ✅ Use in proprietary applications
- ⚠️ Include a copy of the license

---

## FAQ

### Q: Is my eye data being sent anywhere?

**A:** No. This application has zero network code. Your camera frames never leave your machine. Not a single byte is transmitted.

### Q: Can I modify this for my own use?

**A:** Yes. The MIT license permits full modification and redistribution. You can audit every line of code.

### Q: Why not use OpenCV or MediaPipe?

**A:** Those are excellent but heavy:
- OpenCV: ~15-20x larger binary, more CPU
- MediaPipe: Requires ML model downloads, internet (sometimes), heavy inference

This library trades some flexibility for **speed, privacy, and simplicity**.

### Q: Can this estimate gaze direction (3D angle)?

**A:** Not yet. It finds the 2D pupil center on-screen. Full gaze estimation requires:
- Eye-to-camera geometry calibration
- Corneal reflection (glint) detection
- Sclera tracking

This is on the roadmap.

### Q: How do I integrate this into my app?

**A:** Currently, it's a standalone application. Future versions will expose a library API. For now, you can:
- Modify `main.rs` to output positions via IPC or network socket
- Embed the `PureRustTracker` struct in your code
- Use the binary as a subprocess and parse stdout

### Q: What if my eyes don't track well?

**A:** Common issues:
1. **Lighting**: Ensure good, even front-lighting
2. **Camera angle**: Position camera at eye level
3. **Glasses**: Some reflective lenses confuse the pupil detection
4. **Eyes too far**: Keep face 12-24 inches from camera
5. **Calibration skip**: If auto-calibration fails, check camera output first

---

## Technical References

The core algorithm draws from established computer vision techniques:

- **Template Matching**: Lewis, J. P. (1995). "Fast normalized cross-correlation"
- **SAD (Sum of Absolute Differences)**: Classical block matching in video codecs
- **Online Learning**: Exponential moving average for gradual adaptation
- **State Machines**: Robust initialization + tracking paradigm (standard in trackers)

This implementation optimizes these for minimal hardware impact.

---

## Acknowledgments

Built with:
- **Rust** for memory safety and performance
- **nokhwa** for cross-platform camera abstraction
- **minifb** for minimal-overhead rendering
- **sysinfo** for performance telemetry

---

**Last Updated**: May 2026  
**Status**: Early Release / Active Development  
**Author**: [piresjoaov](https://github.com/piresjoaov)
