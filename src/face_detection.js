import * as tf from '@tensorflow/tfjs';
import postprocess from './postprocess';

const MAX_BOXES = 20;
const INPUT_SIZE = 416;
const SCORE_THRESHOLD = .5;
const IOU_THRESHOLD = .3;

async function _loadModel() {
  return await tf.loadGraphModel("https://raw.githubusercontent.com/miudit/tfjs-yolo-demo/master/dist/model/model.json");
}

async function _predict(
  model,
  image,
  maxBoxes,
  scoreThreshold,
  iouThreshold,
  numClasses,
  anchors,
  classNames,
  inputSize,
) {
  //////
  /*
  let outputs = tf.tidy(() => {
    const canvas = document.createElement('canvas');
    canvas.width = inputSize;
    canvas.height = inputSize;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0, inputSize, inputSize);

    let imageTensor = tf.browser.fromPixels(canvas, 3);
    imageTensor = imageTensor.expandDims(0).toFloat().div(tf.scalar(255));

    const outputs = model.executeAsync(imageTensor);
    return outputs;
  });
  */

  const canvas = document.createElement('canvas');
  canvas.width = inputSize;
  canvas.height = inputSize;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0, inputSize, inputSize);

  let imageTensor = tf.browser.fromPixels(canvas, 3);
  imageTensor = imageTensor.expandDims(0).toFloat().div(tf.scalar(255));

  let outputs = await model.executeAsync(imageTensor);

  ///////

  const boxes = await postprocess(
    outputs,
    anchors,
    numClasses,
    classNames,
    image.constructor.name === 'HTMLVideoElement' ?
      [image.videoHeight, image.videoWidth] :
      [image.height, image.width],
    maxBoxes,
    scoreThreshold,
    iouThreshold
  );

  tf.dispose(outputs);

  return boxes;
}

async function detector() {
  console.log("load!")
  let model = await _loadModel();
  //model.summary()
  console.log("load done")

  let classes = ['face', 'background']

  return {
    predict: async function (
      image,
      {
        maxBoxes = MAX_BOXES,
        scoreThreshold = SCORE_THRESHOLD,
        iouThreshold = IOU_THRESHOLD,
        numClasses = classes.length,
        anchors = [10,14, 23,27, 37,58, 81,82, 135,169, 344,319],
        classNames = classes,
        inputSize = INPUT_SIZE,
      } = {}
    ) {
      return await _predict(
        model,
        image,
        maxBoxes,
        scoreThreshold,
        iouThreshold,
        numClasses,
        anchors,
        classNames,
        inputSize,
      );
    },
    dispose: () => {
      model.dispose();
      model = null;
    }
  }
}

const face_detection = {
  detector
};

export default face_detection;