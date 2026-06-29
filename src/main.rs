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
// 2. A IMPLEMENTAÇÃO OPENCV (OTIMIZADA)
// =====================================================================
pub struct OpenCvTracker {
    face_cascade: objdetect::CascadeClassifier,
    eye_cascade: objdetect::CascadeClassifier,
}

impl OpenCvTracker {
    pub fn new(face_path: &str, eye_path: &str) -> opencv::Result<Self> {
        let face_cascade = objdetect::CascadeClassifier::new(face_path)?;
        let eye_cascade = objdetect::CascadeClassifier::new(eye_path)?;
        Ok(Self { face_cascade, eye_cascade })
    }
}

impl VisionEngine for OpenCvTracker {
    fn detect_eyes(&mut self, frame: &Mat) -> opencv::Result<Vec<Point>> {
        let mut gray = Mat::default();
        imgproc::cvt_color(frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
        imgproc::equalize_hist(&gray.clone(), &mut gray)?;

        // 1. Acha o ROSTO primeiro
        let mut faces = Vector::<Rect>::new();
        self.face_cascade.detect_multi_scale(
            &gray,
            &mut faces,
            1.3, // Escala maior = mais rápido (1.3 é ótimo para rostos)
            5,
            0,
            Size::new(100, 100), // Rosto precisa ser grandinho
            Size::new(0, 0),
        )?;

        let mut eye_centers = Vec::new();

        // Se não achou rosto, retorna vazio (Zero falso positivos na parede)
        if faces.is_empty() {
            return Ok(eye_centers);
        }

        // Pega apenas o primeiro rosto detectado (o principal)
        let face = faces.get(0)?;

        // 2. Cria uma "Região de Interesse" (ROI) na METADE SUPERIOR do rosto
        // Isso evita que narinas e boca sejam confundidas com olhos
        let eye_region = Rect::new(
            face.x, 
            face.y + (face.height / 5), // Corta um pouco a testa
            face.width, 
            face.height / 2 // Pega só até a metade do rosto
        );
        
        // Recorta a imagem em tons de cinza apenas para essa região
        let face_roi = Mat::roi(&gray, eye_region)?;

        // 3. Procura os olhos APENAS dentro dessa pequena região cortada
        let mut eyes = Vector::<Rect>::new();
        self.eye_cascade.detect_multi_scale(
            &face_roi,
            &mut eyes,
            1.1,
            5, // Exige mais certeza para evitar falsos olhos no rosto
            0,
            Size::new(20, 20),
            Size::new(0, 0),
        )?;

        // Pega no máximo 2 olhos e traduz as coordenadas de volta para a tela cheia
        for eye in eyes.iter().take(2) {
            let center_x = eye_region.x + eye.x + eye.width / 2;
            let center_y = eye_region.y + eye.y + eye.height / 2;
            eye_centers.push(Point::new(center_x, center_y));
        }

        Ok(eye_centers)
    }
}

// =====================================================================
// 3. O LOOP PRINCIPAL
// =====================================================================
fn main() -> opencv::Result<()> {
    // === A MÁGICA PARA A "BATATA" ===
    // Trava o OpenCV para usar apenas 1 ou 2 threads. 
    // Adeus uso de CPU em 700%!
    core::set_num_threads(2)?;

    println!("Iniciando Motor Visual (OpenCV Mode)...");

    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    if !videoio::VideoCapture::is_opened(&cam)? {
        panic!("Não foi possível abrir a câmera!");
    }

    let mut tracker = OpenCvTracker::new("haarcascade_frontalface_default.xml", "haarcascade_eye.xml")
        .expect("Arquivos XML não encontrados na raiz!");

    let window_name = "Rust Eye Tracker - Limpo";
    highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE)?;

    let mut sys = System::new();
    let pid = sysinfo::get_current_pid().expect("Falha ao capturar PID");
    
    let mut frames = 0;
    let mut last_print = Instant::now();
    let mut frame = Mat::default();

    loop {
        cam.read(&mut frame)?;
        if frame.size()?.width == 0 {
            continue;
        }

        // Se quiser otimizar MAIS AINDA, você pode redimensionar o 'frame'
        // aqui antes de passar pro tracker usando imgproc::resize.

        let eyes = tracker.detect_eyes(&frame)?;

        // Desenha os pontos vermelhos
        for eye in eyes {
            let red = Scalar::new(0.0, 0.0, 255.0, 0.0);
            imgproc::circle(&mut frame, eye, 6, red, -1, imgproc::LINE_8, 0)?;
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
                println!("[Telemetria OpenCV] FPS: {} | CPU: {:.1}% | RAM: {} MB", frames, cpu_usage, ram_mb);
            }
            frames = 0;
            last_print = Instant::now();
        }
    }

    Ok(())
}