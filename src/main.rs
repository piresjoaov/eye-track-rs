mod camera;

use minifb::{Window, WindowOptions};
use std::time::{Duration, Instant};
use sysinfo::System;

// =====================================================================
// 1. THE VISION ENGINE (Blink Survival, Drift and Auto-Recover)
// =====================================================================

#[derive(PartialEq)]
enum TrackerState {
    Calibrating,
    Tracking,
}

struct EyeTracker {
    template: Vec<u8>,
    anchor_template: Vec<u8>, // NEW: The immutable original photo (The Lifesaver)
    template_w: usize,
    template_h: usize,
    last_x: usize,
    last_y: usize,
    frames_lost: usize,
}

impl EyeTracker {
    fn new(w: usize, h: usize) -> Self {
        Self { 
            template: vec![], 
            anchor_template: vec![],
            template_w: w, 
            template_h: h, 
            last_x: 0, 
            last_y: 0,
            frames_lost: 0 
        }
    }

    fn find_pupil_center(&self, gray_buffer: &[u8], width: usize, box_cx: usize, box_cy: usize, box_w: usize, box_h: usize) -> (usize, usize) {
        let start_x = box_cx.saturating_sub(box_w / 2);
        let start_y = box_cy.saturating_sub(box_h / 2);
        
        let mut min_val = 255u8;
        
        // 1. Find the black hole (darkest point in the strict box)
        for y in start_y..(start_y + box_h) {
            for x in start_x..(start_x + box_w) {
                let val = gray_buffer[y * width + x];
                if val < min_val { min_val = val; }
            }
        }

        // 2. Tight tolerance (only the 15 shades closest to the absolute black hole)
        let threshold = min_val.saturating_add(15);
        
        let mut sum_x = 0;
        let mut sum_y = 0;
        let mut count = 0;

        // 3. Pupil Gravitational Center
        for y in start_y..(start_y + box_h) {
            for x in start_x..(start_x + box_w) {
                if gray_buffer[y * width + x] <= threshold {
                    sum_x += x;
                    sum_y += y;
                    count += 1;
                }
            }
        }

        if count > 0 {
            (sum_x / count, sum_y / count)
        } else {
            (box_cx, box_cy) // Failsafe
        }
    }

    fn capture_template(&mut self, gray_buffer: &[u8], width: usize, height: usize, center_x: usize, center_y: usize) {
        self.last_x = center_x;
        self.last_y = center_y;
        self.frames_lost = 0;
        self.template.clear();
        
        let half_w = self.template_w / 2;
        let half_h = self.template_h / 2;

        for ty in 0..self.template_h {
            for tx in 0..self.template_w {
                let px = center_x + tx - half_w;
                let py = center_y + ty - half_h;
                let val = if px < width && py < height { gray_buffer[py * width + px] } else { 0 };
                self.template.push(val);
            }
        }
        // Save the anchor identical to the initial template
        self.anchor_template = self.template.clone();
    }

    fn update_position(&mut self, gray_buffer: &[u8], width: usize, height: usize) -> (usize, usize, usize) {
        let search_range = 15_isize; 
        
        let mut min_sad = u32::MAX;
        let mut best_dx = 0_isize;
        let mut best_dy = 0_isize;

        let half_w = (self.template_w / 2) as isize;
        let half_h = (self.template_h / 2) as isize;

        for dy in -search_range..=search_range {
            for dx in -search_range..=search_range {
                let mut current_sad = 0u32;
                
                for ty in 0..self.template_h {
                    for tx in 0..self.template_w {
                        let screen_x = (self.last_x as isize + dx + tx as isize - half_w) as usize;
                        let screen_y = (self.last_y as isize + dy + ty as isize - half_h) as usize;
                        
                        if screen_x >= width || screen_y >= height { continue; }

                        let screen_pixel = gray_buffer[screen_y * width + screen_x];
                        let template_pixel = self.template[ty * self.template_w + tx];
                        
                        current_sad += screen_pixel.abs_diff(template_pixel) as u32;
                    }
                }

                if current_sad < min_sad {
                    min_sad = current_sad;
                    best_dx = dx;
                    best_dy = dy;
                }
            }
        }

        // Blink limit (The image completely disappeared temporarily)
        let max_acceptable_sad = (self.template_w * self.template_h * 35) as u32;

        if min_sad > max_acceptable_sad {
            self.frames_lost += 1; // Blinked or moved fast. Freeze position.
        } else {
            self.frames_lost = 0;
            self.last_x = (self.last_x as isize + best_dx) as usize;
            self.last_y = (self.last_y as isize + best_dy) as usize;
            
            // --- THE ANCHOR MAGIC (ANTI-DRIFT) ---
            let mut drift_sad = 0u32;
            for ty in 0..self.template_h {
                for tx in 0..self.template_w {
                    let px = self.last_x as isize + tx as isize - half_w;
                    let py = self.last_y as isize + ty as isize - half_h;

                    if px < 0 || py < 0 {
                        continue;
                    }

                    let px = px as usize;
                    let py = py as usize;

                    if px < width && py < height {
                        let screen_pixel = gray_buffer[py * width + px];
                        let anchor_pixel = self.anchor_template[ty * self.template_w + tx];
                        drift_sad += screen_pixel.abs_diff(anchor_pixel) as u32;
                    }
                }
            }

            // If the difference to the original anchor is greater than 45 shades on average, we drifted to the eyebrow/hair.
            let max_drift_sad = (self.template_w * self.template_h * 45) as u32;

            if drift_sad > max_drift_sad {
                // SILENT RESET: Pulls the template back to the original pupil without interrupting tracking
                self.template = self.anchor_template.clone();
            } else {
                // Safe to learn. Rate reduced to 0.5% (Fast Healing)
                self.blend_template(gray_buffer, width, height, 0.005);
            }
        }

        (self.last_x, self.last_y, self.frames_lost)
    }

    fn blend_template(&mut self, gray_buffer: &[u8], width: usize, height: usize, alpha: f32) {
        let half_w = self.template_w as isize / 2;
        let half_h = self.template_h as isize / 2;

        for ty in 0..self.template_h {
            for tx in 0..self.template_w {
                let px = self.last_x as isize + tx as isize - half_w;
                let py = self.last_y as isize + ty as isize - half_h;

                if px < 0 || py < 0 {
                    continue;
                }

                let px = px as usize;
                let py = py as usize;
                
                if px < width && py < height {
                    let new_pixel = gray_buffer[py * width + px] as f32;
                    let old_pixel = self.template[ty * self.template_w + tx] as f32;
                    let blended = (old_pixel * (1.0 - alpha)) + (new_pixel * alpha);
                    self.template[ty * self.template_w + tx] = blended as u8;
                }
            }
        }
    }
}

pub struct PureRustTracker {
    state: TrackerState,
    calibration_frames: usize,
    left_eye: EyeTracker,
    right_eye: EyeTracker,
}

impl PureRustTracker {
    pub fn new() -> Self {
        Self {
            state: TrackerState::Calibrating,
            calibration_frames: 0,
            left_eye: EyeTracker::new(40, 40),
            right_eye: EyeTracker::new(40, 40),
        }
    }

    pub fn detect(&mut self, gray_buffer: &[u8], width: usize, height: usize) -> Option<((usize, usize), (usize, usize))> {
        let face_cx = width / 2;
        let face_cy = height / 2;
        
        let eye_search_w = 60;
        let eye_search_h = 30;
        
        let left_search_cx = face_cx - 50;
        let left_search_cy = face_cy - 40;
        
        let right_search_cx = face_cx + 50;
        let right_search_cy = face_cy - 40;

        match self.state {
            TrackerState::Calibrating => {
                self.calibration_frames += 1;
                
                if self.calibration_frames > 90 {
                    println!("Auto-Snap! Finding pupil in the micro-funnel...");
                    
                    let exact_left = self.left_eye.find_pupil_center(gray_buffer, width, left_search_cx, left_search_cy, eye_search_w, eye_search_h);
                    let exact_right = self.right_eye.find_pupil_center(gray_buffer, width, right_search_cx, right_search_cy, eye_search_w, eye_search_h);

                    self.left_eye.capture_template(gray_buffer, width, height, exact_left.0, exact_left.1);
                    self.right_eye.capture_template(gray_buffer, width, height, exact_right.0, exact_right.1);
                    
                    self.state = TrackerState::Tracking;
                    Some((exact_left, exact_right))
                } else {
                    None 
                }
            }
            TrackerState::Tracking => {
                let (lx, ly, lost_l) = self.left_eye.update_position(gray_buffer, width, height);
                let (rx, ry, lost_r) = self.right_eye.update_position(gray_buffer, width, height);
                
                // If any eye went blind/lost for more than 45 frames (1.5 seconds), we restart the process.
                if lost_l > 45 || lost_r > 45 {
                    println!("Tracking lost! Returning to Macro Calibration...");
                    self.state = TrackerState::Calibrating;
                    self.calibration_frames = 0;
                    return None;
                }

                Some(((lx, ly), (rx, ry)))
            }
        }
    }
}

// =====================================================================
// 2. STABILIZER AND HELPERS (UI)
// =====================================================================

struct Stabilizer {
    smoothed_x: f32,
    smoothed_y: f32,
    alpha: f32,
    initialized: bool,
}

impl Stabilizer {
    fn new(alpha: f32) -> Self {
        Self { smoothed_x: 0.0, smoothed_y: 0.0, alpha, initialized: false }
    }

    fn update(&mut self, raw_x: usize, raw_y: usize) -> (usize, usize) {
        if !self.initialized {
            self.smoothed_x = raw_x as f32;
            self.smoothed_y = raw_y as f32;
            self.initialized = true;
        } else {
            self.smoothed_x = (self.alpha * raw_x as f32) + ((1.0 - self.alpha) * self.smoothed_x);
            self.smoothed_y = (self.alpha * raw_y as f32) + ((1.0 - self.alpha) * self.smoothed_y);
        }
        (self.smoothed_x.round() as usize, self.smoothed_y.round() as usize)
    }
}

fn draw_dot(window_buffer: &mut [u32], width: usize, height: usize, cx: usize, cy: usize, color: u32) {
    let radius = 3;
    let start_x = cx.saturating_sub(radius);
    let end_x = (cx + radius).min(width - 1);
    let start_y = cy.saturating_sub(radius);
    let end_y = (cy + radius).min(height - 1);

    for y in start_y..=end_y {
        for x in start_x..=end_x {
            window_buffer[y * width + x] = color;
        }
    }
}

fn draw_box(window_buffer: &mut [u32], width: usize, height: usize, cx: usize, cy: usize, w: usize, h: usize, color: u32) {
    let half_w = w / 2;
    let half_h = h / 2;
    let start_x = cx.saturating_sub(half_w);
    let end_x = (cx + half_w).min(width - 1);
    let start_y = cy.saturating_sub(half_h);
    let end_y = (cy + half_h).min(height - 1);

    for i in start_x..=end_x {
        window_buffer[start_y * width + i] = color;
        window_buffer[end_y * width + i] = color;
    }
    for j in start_y..=end_y {
        window_buffer[j * width + start_x] = color;
        window_buffer[j * width + end_x] = color;
    }
}

// =====================================================================
// 3. THE MAIN LOOP
// =====================================================================

fn main() {
    println!("Initializing RUST-PURE Eye Tracker Engine (Funnel Mode + Anchor Anti-Drift)...");

    let mut stream = camera::VideoStream::new();
    let width = stream.resolution.width() as usize;
    let height = stream.resolution.height() as usize;

    let mut window_buffer: Vec<u32> = vec![0; width * height];
    let mut gray_buffer: Vec<u8> = vec![0; width * height];
    
    let mut tracker = PureRustTracker::new();
    let mut left_stb = Stabilizer::new(0.4);
    let mut right_stb = Stabilizer::new(0.4);

    let mut window = Window::new("Rust Pure Eye Tracker", width, height, WindowOptions::default()).unwrap();
    window.limit_update_rate(Some(Duration::from_micros(33333))); // ~30 FPS

    let mut sys = System::new();
    let pid = sysinfo::get_current_pid().expect("Failed to get PID");
    let mut frames = 0;
    let mut last_print = Instant::now();

    while window.is_open() && !window.is_key_down(minifb::Key::Escape) {
        let frame = stream.get_frame();
        let raw_yuyv = frame.buffer();

        for (i, macropixel) in raw_yuyv.chunks_exact(4).enumerate() {
            let y0 = macropixel[0]; 
            let y1 = macropixel[2]; 

            gray_buffer[i * 2] = y0;
            gray_buffer[i * 2 + 1] = y1;

            let y0_32 = y0 as u32;
            let y1_32 = y1 as u32;
            window_buffer[i * 2] = (y0_32 << 16) | (y0_32 << 8) | y0_32;
            window_buffer[i * 2 + 1] = (y1_32 << 16) | (y1_32 << 8) | y1_32;
        }

        if tracker.state == TrackerState::Calibrating {
            let face_cx = width / 2;
            let face_cy = height / 2;
            
            draw_box(&mut window_buffer, width, height, face_cx, face_cy, 200, 240, 0x00_00_FF_00); // Face
            draw_box(&mut window_buffer, width, height, face_cx - 50, face_cy - 40, 60, 30, 0x00_FF_00_00); // Left Eye
            draw_box(&mut window_buffer, width, height, face_cx + 50, face_cy - 40, 60, 30, 0x00_FF_00_00); // Right Eye
        }

        if let Some(((lx, ly), (rx, ry))) = tracker.detect(&gray_buffer, width, height) {
            let (slx, sly) = left_stb.update(lx, ly);
            let (srx, sry) = right_stb.update(rx, ry);
            
            if tracker.state == TrackerState::Tracking {
                draw_dot(&mut window_buffer, width, height, slx, sly, 0x00_FF_00_00);
                draw_dot(&mut window_buffer, width, height, srx, sry, 0x00_FF_00_00);
            }
        } else {
            left_stb.initialized = false;
            right_stb.initialized = false;
        }

        window.update_with_buffer(&window_buffer, width, height).unwrap();

        frames += 1;
        if last_print.elapsed().as_secs() >= 1 {
            sys.refresh_processes();
            if let Some(process) = sys.process(pid) {
                let ram_mb = process.memory() / 1024 / 1024;
                let cpu_usage = process.cpu_usage();
                println!("[Pure Telemetry] FPS: {} | CPU: {:.1}% | RAM: {} MB", frames, cpu_usage, ram_mb);
            }
            frames = 0;
            last_print = Instant::now();
        }
    }
}