let utils = new Utils('errorMessage');



//utils.loadCode('indexCode', 'codeEditor');

let streaming = false;
let videoInput = document.getElementById('videoInput');
let startAndStop = document.getElementById('startAndStop');
let canvasOutput = document.getElementById('canvasOutput');
let canvasContext = canvasOutput.getContext('2d');

startAndStop.addEventListener('click', () => {
    if (!streaming) {
        utils.clearError();
        utils.startCamera('qvga', onVideoStarted, 'videoInput');
    } else {
        utils.stopCamera();
        onVideoStopped();
    }
});

function onVideoStarted() {
    streaming = true;
    startAndStop.innerText = 'Stop';
    videoInput.width = videoInput.videoWidth;
    videoInput.height = videoInput.videoHeight;


    startVideoProcess();
}

function onVideoStopped() {
    streaming = false;
    canvasContext.clearRect(0, 0, canvasOutput.width, canvasOutput.height);
    startAndStop.innerText = 'Start';
}

utils.loadOpenCv(() => {
    startAndStop.removeAttribute('disabled');
});




import * as THREE from 'three';

const scene = new THREE.Scene();

let threeWidth = 640;
let threeHeight = 480;

const camera = new THREE.PerspectiveCamera(75, threeWidth/threeHeight, 0.6, 1200);
camera.position.z += 3;

const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setClearColor("#03051a");
renderer.setSize(threeWidth, threeHeight);
document.body.appendChild(renderer.domElement);

window.addEventListener('resize', () => {
    renderer.setSize(threeWidth, threeHeight);
    camera.aspect = threeWidth / threeHeight;
    camera.updateProjectionMatrix();
})

const geometry = new THREE.BoxGeometry(1, 1, 1);
const material = new THREE.MeshBasicMaterial({color: 0xff00000});
const box = new THREE.Mesh(geometry, material);
scene.add(box);

let video, src, dst, gray, cap, classifier, faces;

function startVideoProcess() {
  video = document.getElementById('videoInput');
  src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
  dst = new cv.Mat(video.height, video.width, cv.CV_8UC4);
  gray = new cv.Mat();
  cap = new cv.VideoCapture(video);
  faces = new cv.RectVector();
  classifier = new cv.CascadeClassifier();
  classifier.load('haarcascade_frontalface_default.xml');
  setTimeout(processFaceVideo, 0);
}
const FPS = 30;

function greyProcessVideo() {
    try {
        if (!streaming) {
            // clean and stop.
            src.delete();
            dst.delete();
            return;
        }
        let begin = Date.now();
        // start processing.
        cap.read(src);
        cv.cvtColor(src, dst, cv.COLOR_RGBA2GRAY);
        cv.imshow('canvasOutput', dst);
        // schedule the next one.
        let delay = 1000/FPS - (Date.now() - begin);
        setTimeout(greyProcessVideo, delay);
    } catch (err) {
        utils.printError(err);
    }
};

let x, y, faceWidth, faceHeight;

function processFaceVideo() {
    try {
        if (!streaming) {
            // clean and stop.
            src.delete();
            dst.delete();
            gray.delete();
            faces.delete();
            classifier.delete();
            return;
        }
        let begin = Date.now();
        // start processing.
        cap.read(src);
        src.copyTo(dst);
        cv.cvtColor(dst, gray, cv.COLOR_RGBA2GRAY, 0);
        // detect faces.
        classifier.detectMultiScale(gray, faces, 1.1, 5, 0);
        // draw faces.
        for (let i = 0; i < faces.size(); ++i) {
            let face = faces.get(i);
            console.log(face);
            let point1 = new cv.Point(face.x, face.y);
            x = face.x;
            y = face.y;
            faceWidth = face.width;
            faceHeight = face.height;
            let point2 = new cv.Point(face.x + face.width, face.y + face.height);
            cv.rectangle(dst, point1, point2, [255, 0, 0, 255]);
        }
        cv.imshow('canvasOutput', dst);
        // schedule the next one.
        let delay = 1000/FPS - (Date.now() - begin);
        setTimeout(processFaceVideo, delay);
    } catch (err) {
        utils.printError(err);
    }
};

const rendering = function() {
    requestAnimationFrame(rendering);
    camera.position.x = x/320 * 6 - 3;
    camera.position.y = (1 - y/240) * 6 - 3;
    let area = 1- (faceWidth * faceHeight)/19200;
    camera.position.z = 2 + area*3;
    camera.lookAt(box.position);
    renderer.render(scene, camera);
}

rendering();

function printXY() {
    console.log("X: ", x, " Y: ", y, " Width: ", faceWidth, " Height: ", faceHeight);
}