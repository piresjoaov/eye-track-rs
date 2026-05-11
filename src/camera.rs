use nokhwa::{
    pixel_format::YuyvFormat,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType, Resolution},
    Camera,
};

pub struct VideoStream {
    camera: Camera,
    pub resolution: Resolution,
}

impl VideoStream {
    pub fn new() -> Self {
        let index = CameraIndex::Index(0);
        // Request the rawest format possible to avoid driver-level overhead
        let format = RequestedFormat::new::<YuyvFormat>(RequestedFormatType::AbsoluteHighestResolution);
        
        let mut camera = Camera::new(index, format).expect("Failed to initialize camera");
        camera.open_stream().expect("Failed to open video stream");
        
        let resolution = camera.camera_format().resolution();
        
        Self { camera, resolution }
    }

    // Returns a raw frame buffer. No heavy abstractions.
    pub fn get_frame(&mut self) -> nokhwa::Buffer {
        self.camera.frame().expect("Failed to capture frame")
    }
}