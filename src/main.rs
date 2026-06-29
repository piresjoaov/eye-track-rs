use opencv::{
    core::{self, Point, Rect, Scalar, Size, Vector, Mat},
    highgui, imgproc, objdetect, prelude::*, videoio,
};
use std::time::Instant;
use sysinfo::System;

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

// =====================================================================
// 3. A IMPLEMENTAÇÃO OPENCV (OTIMIZADA PARA "BATATA")
// =====================================================================
pub struct OpenCvTracker {
    face_cascade: objdetect::CascadeClassifier,
    eye_cascade: objdetect::CascadeClassifier,
    last_face: Option<Rect>,
    frame_count: i32,
    detect_face_every: i32,
}

impl OpenCvTracker {
    pub fn new(face_path: &str, eye_path: &str) -> opencv::Result<Self> {
        let face_cascade = objdetect::CascadeClassifier::new(face_path)?;
        let eye_cascade = objdetect::CascadeClassifier::new(eye_path)?;
        Ok(Self { 
            face_cascade, 
            eye_cascade,
            last_face: None,
            frame_count: 0,
            detect_face_every: 3, // Detecta rosto a cada 3 frames (reutiliza nos outros)
        })
    }
}

impl VisionEngine for OpenCvTracker {
    fn detect_eyes(&mut self, frame: &Mat) -> opencv::Result<Vec<Point>> {
        let mut gray = Mat::default();
        imgproc::cvt_color(frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
        
        // ========== OTIMIZAÇÃO: REMOVER EQUALIZED DESNECESSÁRIA ==========
        // Não usava equalized_hist porque detectamos em &gray, não em &equalized
        // Isso economiza ~0.5 a 1 MB de alocação desnecessária

        let mut face = None;

        // ========== OTIMIZAÇÃO 1: REUTILIZAR DETECÇÃO DE ROSTO ==========
        // Só detecta a cada N frames, reutiliza o anterior
        if self.frame_count % self.detect_face_every == 0 {
            let mut faces = Vector::<Rect>::new();
            self.face_cascade.detect_multi_scale(
                &gray,
                &mut faces,
                1.5,    // ← OTIMIZAÇÃO: Escala maior (1.5 em vez de 1.3)
                5,
                0,
                Size::new(120, 120), // ← OTIMIZAÇÃO: Face mínima maior
                Size::new(0, 0),
            )?;

            if !faces.is_empty() {
                let detected_face = faces.get(0)?;
                self.last_face = Some(detected_face);
                face = Some(detected_face);
            }
        } else {
            // Reutiliza o rosto anterior se disponível
            face = self.last_face;
        }

        self.frame_count = (self.frame_count + 1) % self.detect_face_every;

        let mut eye_centers = Vec::new();

        // Se não achou rosto, retorna vazio
        if face.is_none() {
            return Ok(eye_centers);
        }

        let face = face.unwrap();

        // ========== OTIMIZAÇÃO 2: ROI AGRESSIVA PARA OLHOS ==========
        let eye_region = Rect::new(
            face.x, 
            face.y + (face.height / 4),  // Corta mais da testa
            face.width, 
            (face.height / 2) + 20       // Ligeiramente maior para garantir captura
        );
        
        let face_roi = Mat::roi(&gray, eye_region)?;

        // ========== OTIMIZAÇÃO 3: SPLIT L/R CONFORME SUA IDEIA ==========
        let mid_x = eye_region.width / 2;

        // Detecta OLHO ESQUERDO
        let left_roi_rect = Rect::new(0, 0, mid_x, eye_region.height);
        let left_roi = Mat::roi(&face_roi, left_roi_rect)?;
        
        let mut left_eyes = Vector::<Rect>::new();
        self.eye_cascade.detect_multi_scale(
            &left_roi,
            &mut left_eyes,
            1.05,  // ← Escala fina para cada metade
            10,    // ← OTIMIZAÇÃO: minNeighbors mais alto (menos falsos positivos)
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

        // Coleta olho esquerdo
        if !left_eyes.is_empty() {
            let eye = left_eyes.get(0)?;
            let center_x = eye_region.x + eye.x + eye.width / 2;
            let center_y = eye_region.y + eye.y + eye.height / 2;
            eye_centers.push(Point::new(center_x, center_y));
        }

        // Coleta olho direito
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
// 4. O LOOP PRINCIPAL COM TELEMETRIA DETALHADA
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

    // ========== OTIMIZAÇÃO 4: REDIMENSIONAR FRAME ==========
    // Detectar em resolução menor = bem mais rápido
    // Você pode ajustar o scale_factor entre 0.5 e 1.0
    let scale_factor = 0.7;  // Processa em 70% da resolução

    let mut tracker = OpenCvTracker::new("haarcascade_frontalface_default.xml", "haarcascade_eye.xml")
        .expect("Arquivos XML não encontrados na raiz!");
    
    print_ram(&mut sys, pid, "Cascade");

    let window_name = "Rust Eye Tracker - Otimizado";
    highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE)?;
    
    print_ram(&mut sys, pid, "Window");

    // Captura primeiro frame para medir
    let mut frame = Mat::default();
    cam.read(&mut frame)?;
    
    print_ram(&mut sys, pid, "Primeiro frame");
    
    println!("\n===== INICIANDO LOOP =====\n");

    let mut frames = 0;
    let mut last_print = Instant::now();
    let mut resized_frame = Mat::default();

    loop {
        cam.read(&mut frame)?;
        if frame.size()?.width == 0 {
            continue;
        }

        // ========== OTIMIZAÇÃO 5: REDIMENSIONAR ANTES DE DETECTAR ==========
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

        // Desenha os pontos vermelhos no frame ORIGINAL para exibição
        for eye in &eyes {
            let red = Scalar::new(0.0, 255.0, 0.0, 0.0);
            imgproc::circle(&mut frame, *eye, 3, red, -1, imgproc::LINE_8, 0)?;
        }

        highgui::imshow(window_name, &frame)?;

        if highgui::wait_key(1)? == 27 {
            break;
        }

        // Telemetria Leve
        frames += 1;
        if last_print.elapsed().as_secs() >= 1 {
            sys.refresh_processes();
            if let Some(process) = sys.process(pid) {
                let ram_mb = process.memory() / 1024 / 1024;
                let cpu_usage = process.cpu_usage();
                println!("[Telemetria] FPS: {} | CPU: {:.1}% | RAM: {} MB", frames, cpu_usage, ram_mb);
            }
            frames = 0;
            last_print = Instant::now();
        }
    }

    Ok(())
}
