/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';

import {ControllerDataset} from './controller_dataset';
import * as ui from './ui';
import {Webcam} from './webcam';

// The number of classes we want to predict. In this example, we will be
// predicting 4 classes for up, down, left, and right.
const NUM_CLASSES = 4;

// A webcam class that generates Tensors from the images from the webcam.
const webcam = new Webcam(document.getElementById('webcam'));

let truncatedMobileNet;
let model;


let isPredicting = false;

async function predict() {
  ui.isPredicting();
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      // Capture the frame from the webcam.
      const img = webcam.capture();

      // Make a prediction through mobilenet, getting the internal activation of
      // the mobilenet model, i.e., "embeddings" of the input images.
      // const embeddings = 

      // Make a prediction through our newly-trained model using the embeddings
      // from mobilenet as input.
      const predictions = truncatedMobileNet.predict(img);

      // Returns the index with the maximum probability. This number corresponds
      // to the class the model thinks is the most probable given the input.
      return predictions.as1D().argMax();
    });

    const classId = (await predictedClass.data())[0];
    predictedClass.dispose();
    document.getElementById('prediction').textContent = classId;
    console.log(classId);
    // ui.predictClass(classId);
    await tf.nextFrame();
  }
  ui.donePredicting();
}

document.getElementById('predict').addEventListener('click', () => {
  isPredicting = true;
  predict();
});

document.getElementById('done').addEventListener('click', () => {
  isPredicting = false;
});

async function init() {
  try {
    await webcam.setup();
  } catch (e) {
    document.getElementById('no-webcam').style.display = 'block';
  }
  const mobilenet = await tf.loadModel('http://192.168.1.156:8887/model.json');

  truncatedMobileNet = mobilenet;
  // console.log(webcam.capture());
  // Warm up the model. This uploads weights to the GPU and compiles the WebGL
  // programs so the first time we collect data from the webcam it will be
  // quick.
  // tf.tidy(() => truncatedMobileNet.predict(webcam.capture()));

  ui.init();
}

// Initialize the application.
init();
