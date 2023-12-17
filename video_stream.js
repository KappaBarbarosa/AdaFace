var video;
var div = null;
var stream;
var captureCanvas;
var imgElement;
var labelElement;
var detection;

var pendingResolve = null;
var shutdown = false;

function removeDom() {
    stream.getVideoTracks()[0].stop();
    video.remove();
    div.remove();
    video = null;
    div = null;
    stream = null;
    imgElement = null;
    captureCanvas = null;
    labelElement = null;
}

function onAnimationFrame() {
    if (!shutdown) {
    window.requestAnimationFrame(onAnimationFrame);
    }
    if (pendingResolve) {
    var result = "";
    if (!shutdown) {
        captureCanvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
        result = captureCanvas.toDataURL('image/jpeg', 0.8)
    }
    var lp = pendingResolve;
    pendingResolve = null;
    lp(result);
    }
}


const { JSDOM } = require('jsdom');
const dom = new JSDOM();
global.document = dom.window.document;

async function createDom() {
    if (div !== null) {
        return stream;
    }

    div = document.createElement('div');
    div.style.border = '2px solid black';
    div.style.padding = '3px';
    div.style.width = '100%';
    div.style.maxWidth = '600px';
    document.body.appendChild(div);

    const modelOut = document.createElement('div');
    modelOut.innerHTML = "<span>Status:</span>";
    labelElement = document.createElement('span');
    labelElement.innerText = 'No data';
    labelElement.style.fontWeight = 'bold';
    modelOut.appendChild(labelElement);
    div.appendChild(modelOut);

    video = document.createElement('video');
    video.style.display = 'block';
    video.width = div.clientWidth - 6;
    video.setAttribute('playsinline', '');
    video.onclick = () => { shutdown = true; };

    // Note: In Node.js, you won't have access to navigator or mediaDevices directly.
    // You'll need to use a library that provides access to camera streams in Node.js.

    div.appendChild(video);

    imgElement = document.createElement('img');
    imgElement.style.position = 'absolute';
    imgElement.style.zIndex = 1;
    imgElement.onclick = () => { shutdown = true; };
    div.appendChild(imgElement);

    const instruction = document.createElement('div');
    instruction.innerHTML =
        '<span style="color: red; font-weight: bold;">' +
        'When finished, click here or on the video to stop this demo</span>';
    div.appendChild(instruction);
    instruction.onclick = () => { shutdown = true; };

    // Use process.stdin to listen for keyboard input
    process.stdin.setRawMode(true);
    process.stdin.resume();
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', function (key) {
        if (key === '\u0003') {
            // Ctrl+C pressed, shutdown the program
            shutdown = true;
        } else {
            // Handle other keypress events as needed
            detection = true;
        }
    });

    // Note: You won't be able to use window.requestAnimationFrame directly in Node.js.
    // You might need to find an alternative method.

    return stream;
}


// async function createDom() {
//     if (div !== null) {
//     return stream;
//     }

//     div = document.createElement('div');
//     div.style.border = '2px solid black';
//     div.style.padding = '3px';
//     div.style.width = '100%';
//     div.style.maxWidth = '600px';
//     document.body.appendChild(div);

//     const modelOut = document.createElement('div');
//     modelOut.innerHTML = "<span>Status:</span>";
//     labelElement = document.createElement('span');
//     labelElement.innerText = 'No data';
//     labelElement.style.fontWeight = 'bold';
//     modelOut.appendChild(labelElement);
//     div.appendChild(modelOut);

//     video = document.createElement('video');
//     video.style.display = 'block';
//     video.width = div.clientWidth - 6;
//     video.setAttribute('playsinline', '');
//     video.onclick = () => { shutdown = true; };
//     stream = await navigator.mediaDevices.getUserMedia(
//         {video: { facingMode: "environment"}});
//     div.appendChild(video);

//     imgElement = document.createElement('img');
//     imgElement.style.position = 'absolute';
//     imgElement.style.zIndex = 1;
//     imgElement.onclick = () => { shutdown = true; };
//     div.appendChild(imgElement);

//     const instruction = document.createElement('div');
//     instruction.innerHTML =
//         '<span style="color: red; font-weight: bold;">' +
//         'When finished, click here or on the video to stop this demo</span>';
//     div.appendChild(instruction);
//     instruction.onclick = () => { shutdown = true; };
//     window.onkeydown=function(){ detection = true; };

//     video.srcObject = stream;
//     await video.play();

//     captureCanvas = document.createElement('canvas');
//     captureCanvas.width = 640; //video.videoWidth;
//     captureCanvas.height = 480; //video.videoHeight;
//     window.requestAnimationFrame(onAnimationFrame);

//     return stream;
// }

async function stream_frame(label, imgData) {
    if (shutdown) {
    removeDom();
    shutdown = false;
    return '';
    }

    var preCreate = Date.now();
    stream = await createDom();

    var preShow = Date.now();
    if (label != "") {
        labelElement.innerHTML = label;
    }
    console.log(imgData)
    if (imgData != undefined){ 
        var videoRect = video.getClientRects()[0];
        imgElement.style.top = videoRect.top + "px";
        imgElement.style.left = videoRect.left + "px";
        imgElement.style.width = videoRect.width + "px";
        imgElement.style.height = videoRect.height + "px";
        imgElement.src = imgData;
    }

    var preCapture = Date.now();
    var result = await new Promise(function(resolve, reject) {
        pendingResolve = resolve;
    });
    shutdown = false;

    return {'create': preShow - preCreate,
            'show': preCapture - preShow,
            'capture': Date.now() - preCapture,
            'img': result};
}

const [,, label, bbox] = process.argv;

stream_frame(label, bbox)
// const [,,func, label, bbox] = process.argv;
// switch (func){
//     case "removeDom":
//         removeDom()
//     case "onAnimationFrame":
//         onAnimationFrame()
//     case "createDom":
//         createDom()
//     case "stream_frame":
//         stream_frame(label, bbox)
// }