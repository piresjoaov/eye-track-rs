use opencv::{
    core::{self, Point, Rect, Scalar, Size, Vector, Mat},
    highgui, imgproc, objdetect, prelude::*, videoio,
};
use std::time::Instant;
use sysinfo::System;
use enigo::{Enigo, MouseControllable};

// =====================================================================
// NOVO: CONTROLADOR DE MOUSE E SUAVIZAÇÃO (EMA)
// =====================================================================
pub struct MouseController {
    enigo: Enigo,
    smoothed_x: f32,
    smoothed_y: f32,
    alpha: f32,        // O fator de suavização (0.0 a 1.0). ~0.15 é um bom começo.
    screen_w: f32,
    screen_h: f32,
}

impl MouseController {
    pub fn new(alpha: f32, screen_w: f32, screen_h: f32) -> Self {
        Self {
            enigo: Enigo::new(),
            smoothed_x: screen_w / 2.0, // Começa no centro da tela
            smoothed_y: screen_h / 2.0,
            alpha,
            screen_w,
            screen_h,
        }
    }

    pub fn update(&mut self, raw_x: f32, raw_y: f32) {
        // 1. O FILTRO EMA: Mistura a nova posição com a antiga
        self.smoothed_x = (self.alpha * raw_x) + ((1.0 - self.alpha) * self.smoothed_x);
        self.smoothed_y = (self.alpha * raw_y) + ((1.0 - self.alpha) * self.smoothed_y);

        // 2. MOVE O MOUSE (Apenas se a mudança for significativa para poupar CPU)
        let _ = self.enigo.mouse_move_to(self.smoothed_x as i32, self.smoothed_y as i32);
    }

    pub fn get_position(&self) -> (i32, i32) {
        (self.smoothed_x as i32, self.smoothed_y as i32)
    }
}

// =====================================================================
// 1. A INTERFACE LIMPA
// =====================================================================
pub trait VisionEngine {
    fn detect_eyes(&mut self, frame: &Mat) -> opencv::Result<Vec<Point>>;
}

// =====================================================================
// 2. UTILITÁRIOS DE DEBUG
// =====================================================================
fn print_ram(sys: &mut System, pid: sysinfo::Pid, label: &str) {
    sys.refresh_processes();
    if let Some(process) = sys.process(pid) {
        let ram_mb = process.memory() / 1024 / 1024;
        println!("[RAM] {} ............ {} MB", label, ram_mb);
    }
}

fn estimate_headless_ram(current_mb: i32, has_display: bool) -> i32 {
    if has_display {
        // Subtrair overhead de HighGUI (~4 MB) + frame buffer de exibição (~8-10 MB)
        (current_mb - 12).max(0)
    } else {
        current_mb
    }
}

// =====================================================================
// 3. A IMPLEMENTAÇÃO OPENCV (ESTRATÉGIA: REUTILIZAR ROI)
// =====================================================================
pub struct OpenCvTracker {
    face_cascade: objdetect::CascadeClassifier,
    eye_cascade: objdetect::CascadeClassifier,
    last_face: Option<Rect>,
    last_eye_region: Option<Rect>, // ← Cache da ROI de detecção de olhos
    frame_count: i32,
    detect_face_every: i32,
    reuse_eye_region_frames: i32, // ← Reutiliza ROI por N frames
}

impl OpenCvTracker {
    pub fn new(face_path: &str, eye_path: &str) -> opencv::Result<Self> {
        let face_cascade = objdetect::CascadeClassifier::new(face_path)?;
        let eye_cascade = objdetect::CascadeClassifier::new(eye_path)?;
        Ok(Self { 
            face_cascade, 
            eye_cascade,
            last_face: None,
            last_eye_region: None,
            frame_count: 0,
            detect_face_every: 3,
            reuse_eye_region_frames: 5,
        })
    }
}

impl VisionEngine for OpenCvTracker {
    fn detect_eyes(&mut self, frame: &Mat) -> opencv::Result<Vec<Point>> {
        let mut gray = Mat::default();
        imgproc::cvt_color(frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

        let mut face = None;

        // ========== OTIMIZAÇÃO 1: REUTILIZAR DETECÇÃO DE ROSTO ==========
        // Detecta rosto a cada N frames, reutiliza nos outros
        if self.frame_count % self.detect_face_every == 0 {
            let mut faces = Vector::<Rect>::new();
            self.face_cascade.detect_multi_scale(
                &gray,
                &mut faces,
                1.5,
                5,
                0,
                Size::new(120, 120),
                Size::new(0, 0),
            )?;

            if !faces.is_empty() {
                let detected_face = faces.get(0)?;
                self.last_face = Some(detected_face);
                face = Some(detected_face);
            }
        } else {
            face = self.last_face;
        }

        self.frame_count = (self.frame_count + 1) % self.detect_face_every;

        let mut eye_centers = Vec::new();

        // Se não achou rosto, retorna vazio
        if face.is_none() {
            return Ok(eye_centers);
        }

        let face = face.unwrap();

        // ========== OTIMIZAÇÃO 2: DEFINIR ROI DE OLHOS ==========
        // Reutiliza a mesma ROI por vários frames ou refaz a cada N frames
        let eye_region = if self.frame_count % self.reuse_eye_region_frames == 0 {
            let new_region = Rect::new(
                face.x, 
                face.y + (face.height / 4),
                face.width, 
                (face.height / 2) + 20
            );
            self.last_eye_region = Some(new_region);
            new_region
        } else {
            self.last_eye_region.unwrap_or_else(|| {
                Rect::new(
                    face.x, 
                    face.y + (face.height / 4),
                    face.width, 
                    (face.height / 2) + 20
                )
            })
        };

        let face_roi = Mat::roi(&gray, eye_region)?;

        // ========== OTIMIZAÇÃO 3: SPLIT L/R PARA DETECTAR 2 OLHOS ==========
        let mid_x = eye_region.width / 2;

        // Detecta OLHO ESQUERDO
        let left_roi_rect = Rect::new(0, 0, mid_x, eye_region.height);
        let left_roi = Mat::roi(&face_roi, left_roi_rect)?;
        
        let mut left_eyes = Vector::<Rect>::new();
        self.eye_cascade.detect_multi_scale(
            &left_roi,
            &mut left_eyes,
            1.05,
            10,
            0,
            Size::new(15, 15),
            Size::new(0, 0),
        )?;

        // Detecta OLHO DIREITO
        let right_roi_rect = Rect::new(mid_x, 0, mid_x, eye_region.height);
        let right_roi = Mat::roi(&face_roi, right_roi_rect)?;
        
        let mut right_eyes = Vector::<Rect>::new();
        self.eye_cascade.detect_multi_scale(
            &right_roi,
            &mut right_eyes,
            1.05,
            10,
            0,
            Size::new(15, 15),
            Size::new(0, 0),
        )?;

        // ========== FIXO: Coleta AMBOS os olhos corretamente ==========
        // Coleta olho ESQUERDO (metade esquerda da face)
        if !left_eyes.is_empty() {
            let eye = left_eyes.get(0)?;
            let center_x = eye_region.x + eye.x + eye.width / 2;
            let center_y = eye_region.y + eye.y + eye.height / 2;
            eye_centers.push(Point::new(center_x, center_y));
        }

        // Coleta olho DIREITO (metade direita da face)
        if !right_eyes.is_empty() {
            let eye = right_eyes.get(0)?;
            let center_x = eye_region.x + mid_x + eye.x + eye.width / 2;
            let center_y = eye_region.y + eye.y + eye.height / 2;
            eye_centers.push(Point::new(center_x, center_y));
        }

        Ok(eye_centers)
    }
}

// =====================================================================
// 4. O LOOP PRINCIPAL COM CONTROLE DE MOUSE
// =====================================================================
fn main() -> opencv::Result<()> {
    // === INÍCIO DA TELEMETRIA DETALHADA ===
    let mut sys = System::new();
    let pid = sysinfo::get_current_pid().expect("Falha ao capturar PID");
    
    println!("\n===== TELEMETRIA DE MEMÓRIA =====\n");
    
    print_ram(&mut sys, pid, "Inicial");

    // === A MÁGICA PARA A "BATATA" ===
    core::set_num_threads(2)?;

    println!("\nIniciando Motor Visual (OpenCV Mode - OTIMIZADO)...");

    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    if !videoio::VideoCapture::is_opened(&cam)? {
        panic!("Não foi possível abrir a câmera!");
    }
    
    print_ram(&mut sys, pid, "Camera");

    // ========== OTIMIZAÇÃO: REDIMENSIONAR FRAME ==========
    let scale_factor = 0.7;

    let mut tracker = OpenCvTracker::new("haarcascade_frontalface_default.xml", "haarcascade_eye.xml")
        .expect("Arquivos XML não encontrados na raiz!");
    
    print_ram(&mut sys, pid, "Cascade");

    let window_name = "Rust Eye Tracker - Otimizado";
    highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE)?;
    
    print_ram(&mut sys, pid, "Window (com display)");

    // Captura primeiro frame para medir
    let mut frame = Mat::default();
    cam.read(&mut frame)?;
    
    print_ram(&mut sys, pid, "Primeiro frame (com display)");

    // ========== ESTIMATIVA DE RAM HEADLESS ==========
    sys.refresh_processes();
    if let Some(process) = sys.process(pid) {
        let current_mb = (process.memory() / 1024 / 1024) as i32;
        let headless_mb = estimate_headless_ram(current_mb, true);
        println!("[ESTIMATIVA] RAM Headless (sem HighGUI): ~{} MB", headless_mb);
    }
    
    // ========== INICIALIZA O CONTROLADOR DE MOUSE ==========
    // IMPORTANTE: Substitua 1920 e 1080 pela resolução real do seu monitor!
    let mut mouse_ctrl = MouseController::new(0.15, 1920.0, 1080.0);
    
    println!("\n===== INICIANDO LOOP =====\n");
    println!("[INFO] Estratégia: Reutilizar ROI de face/olhos por 3-5 frames");
    println!("[INFO] MOUSE CONTROL ATIVADO - Pressione ESC para sair\n");

    let mut frames = 0;
    let mut last_print = Instant::now();
    let mut resized_frame = Mat::default();

    loop {
        cam.read(&mut frame)?;
        if frame.size()?.width == 0 {
            continue;
        }

        // ========== OTIMIZAÇÃO: REDIMENSIONAR ANTES DE DETECTAR ==========
        let new_width = (frame.size()?.width as f32 * scale_factor) as i32;
        let new_height = (frame.size()?.height as f32 * scale_factor) as i32;
        
        imgproc::resize(
            &frame,
            &mut resized_frame,
            Size::new(new_width, new_height),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;

        // Detecta nos frame redimensionado
        let mut eyes = tracker.detect_eyes(&resized_frame)?;

        // Escala coordenadas de volta para o frame original
        let inv_scale = 1.0 / scale_factor;
        for eye in &mut eyes {
            eye.x = (eye.x as f32 * inv_scale) as i32;
            eye.y = (eye.y as f32 * inv_scale) as i32;
        }

        // ========== NOVO: LÓGICA DO MOUSE ==========
        if eyes.len() == 2 {
            // Se achou os dois olhos, tira a média (ponto entre o nariz)
            let gaze_x = (eyes[0].x + eyes[1].x) as f32 / 2.0;
            let gaze_y = (eyes[0].y + eyes[1].y) as f32 / 2.0;

            // --- CALIBRAÇÃO (Ajuste esses valores testando na sua webcam) ---
            // Qual é a área (x, y) da câmera que representa as pontas da sua tela?
            // DICA: Olhe para cada canto da tela e anote a posição do seu olho
            let cam_min_x = 150.0; // Ponto x quando você olha pra ponta ESQUERDA da tela
            let cam_max_x = 450.0; // Ponto x quando você olha pra ponta DIREITA
            let cam_min_y = 100.0; // Ponto y quando olha pro TOPO
            let cam_max_y = 300.0; // Ponto y quando olha pro CHÃO

            // 1. Normaliza o valor da câmera para uma porcentagem (0.0 a 1.0)
            let mut pct_x = (gaze_x - cam_min_x) / (cam_max_x - cam_min_x);
            let mut pct_y = (gaze_y - cam_min_y) / (cam_max_y - cam_min_y);

            // Garante que não ultrapassa os limites da tela
            pct_x = pct_x.clamp(0.0, 1.0);
            pct_y = pct_y.clamp(0.0, 1.0);

            // 2. Mapeia a porcentagem para os pixels do monitor (ex: 1920x1080)
            // OBS: A câmera espelha a imagem, então invertemos o eixo X (1.0 - pct_x)
            let target_screen_x = (1.0 - pct_x) * mouse_ctrl.screen_w;
            let target_screen_y = pct_y * mouse_ctrl.screen_h;

            // 3. Suaviza e move o mouse!
            mouse_ctrl.update(target_screen_x, target_screen_y);
        }

        // Desenha os pontos vermelhos no frame ORIGINAL para exibição
        for (idx, eye) in eyes.iter().enumerate() {
            let red = Scalar::new(0.0, 255.0, 0.0, 0.0);
            imgproc::circle(&mut frame, *eye, 5, red, -1, imgproc::LINE_8, 0)?;
            
            // Desenha o índice do olho
            imgproc::put_text(
                &mut frame,
                &format!("{}", idx),
                Point::new(eye.x + 10, eye.y - 10),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.5,
                red,
                1,
                imgproc::LINE_8,
                false,
            )?;
        }

        // ========== DEBUG: Mostra status e posição do mouse ==========
        let status_text = format!("Olhos: {} | Mouse: {:?}", eyes.len(), mouse_ctrl.get_position());
        let text_color = if eyes.len() == 2 {
            Scalar::new(0.0, 255.0, 0.0, 0.0) // Verde (2 olhos)
        } else if eyes.len() == 1 {
            Scalar::new(0.0, 165.0, 255.0, 0.0) // Laranja (1 olho)
        } else {
            Scalar::new(255.0, 0.0, 0.0, 0.0) // Vermelho (0 olhos)
        };

        imgproc::put_text(
            &mut frame,
            &status_text,
            Point::new(10, 30),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.7,
            text_color,
            2,
            imgproc::LINE_8,
            false,
        )?;

        highgui::imshow(window_name, &frame)?;

        if highgui::wait_key(1)? == 27 {
            break;
        }

        // Telemetria Leve
        frames += 1;
        if last_print.elapsed().as_secs() >= 1 {
            sys.refresh_processes();
            if let Some(process) = sys.process(pid) {
                let current_mb = (process.memory() / 1024 / 1024) as i32;
                let headless_mb = estimate_headless_ram(current_mb, true);
                let cpu_usage = process.cpu_usage();
                
                println!(
                    "[Telemetria] FPS: {} | CPU: {:.1}% | RAM: {} MB (Headless: ~{} MB)",
                    frames, cpu_usage, current_mb, headless_mb
                );
            }
            frames = 0;
            last_print = Instant::now();
        }
    }

    Ok(())
}
