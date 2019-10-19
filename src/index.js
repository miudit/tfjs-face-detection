if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker
      .register('sw.js')
      .then(reg => console.log('Service Worker: Registered (Pages)'))
      .catch(err => console.log(`Service Worker: Error: ${err}`));
  });
}

import './styles/index.scss';
import * as tf from '@tensorflow/tfjs';
import face_detection from './face_detection';

const loader = document.getElementById('loader');
const spinner = document.getElementById('spinner');
const webcam = document.getElementById('webcam');
const wrapper = document.getElementById('webcam-wrapper');
const rects = document.getElementById('rects');
const load_button = document.getElementById('load');
const predict_button = document.getElementById('predict');

let model;

(async function main() {
  try {
    await setupWebCam();

    load_button.addEventListener('click', () => load(load_button));
    let threshold = .3;
    predict_button.addEventListener('click', () => predict(threshold));
  } catch (e) {
    console.error(e);
  }
})();

async function setupWebCam() {
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    const stream = await navigator.mediaDevices.getUserMedia({
      'audio': false,
      'video': { facingMode: 'environment' }
    });
    window.stream = stream;
    webcam.srcObject = stream;
  }
}

async function load(button) {
  if (model) {
    model.dispose();
    model = null;
  }

  rects.innerHTML = '';
  loader.style.display = 'block';
  spinner.style.display = 'block';
  setButtons(button);

  setTimeout(async () => {
    progress(6);
    model = await face_detection.detector();
  }, 200);
}

function setButtons(button) {
  load_button.className = '';
  button.className = 'selected';
}

function progress(totalModel) {
  let cnt = 0;
  Promise.all = (all => {
    return function then(reqs) {
      if (reqs.length === totalModel && cnt < totalModel * 2)
        reqs.map(req => {
          return req.then(r => {
            loader.setAttribute('percent', (++cnt / totalModel * 50).toFixed(1));
            if (cnt === totalModel * 2) {
              loader.style.display = 'none';
              spinner.style.display = 'none';
              loader.setAttribute('percent', '0.0');
            }
          });
        });
      return all.apply(this, arguments);
    }
  })(Promise.all);
}

async function run() {
  let interval = 100
  if (_interval) {
    interval = _interval.options[_interval.selectedIndex].value;
  }
  console.log("interval = " + interval)
  if (model) {
    let threshold = .3;
    await predict(threshold);
  }
  setTimeout(run, interval);
}

async function predict(threshold) {
  console.log(`Start with ${tf.memory().numTensors} tensors`);

  const start = performance.now();
  const boxes = await model.predict(webcam, { scoreThreshold: threshold });
  const end = performance.now();

  console.log(`Inference took ${end - start} ms`);
  console.log(`End with ${tf.memory().numTensors} tensors`);
  calc_time.innerHTML = `計算時間；${end - start} ms`

  drawBoxes(boxes);

  setTimeout(predict, 100, threshold)
}

let colors = {};

function drawBoxes(boxes) {
  console.log(boxes);
  rects.innerHTML = '';

  const cw = Math.abs(webcam.clientWidth);
  const ch = Math.abs(webcam.clientHeight);
  const vw = Math.abs(webcam.videoWidth);
  const vh = Math.abs(webcam.videoHeight);

  console.log("cw = ", cw)
  console.log("ch = ", ch)
  console.log("vw = ", vw)
  console.log("vh = ", vh)

  const scaleW = cw / vw;
  const scaleH = ch / vh;

  console.log("scaleW = ", scaleW)
  console.log(scaleW)
  console.log("scaleH = ", scaleH)
  console.log(scaleH)

  wrapper.style.width = `${cw}px`;
  wrapper.style.height = `${ch}px`;

  boxes.map((box) => {
    if (!(box['class'] in colors)) {
      colors[box['class']] = '#' + Math.floor(Math.random() * 16777215).toString(16);
    }

    const rect = document.createElement('div');
    rect.className = 'rect';
    rect.style.top = `${box['top'] * scaleH}px`;
    rect.style.left = `${box['left'] * scaleW}px`;
    rect.style.width = `${box['width'] * scaleW - 4}px`;
    rect.style.height = `${box['height'] * scaleH - 4}px`;
    rect.style.borderColor = colors[box['class']];

    const text = document.createElement('div');
    text.className = 'text';
    text.innerText = `${box['class']} ${box['score'].toFixed(2)}`;
    text.style.color = colors[box['class']];

    rect.appendChild(text);
    rects.appendChild(rect);
  });
}