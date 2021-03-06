let model;

var tamilCharacters = ['க','ச','ட','த','ப','ற','ங','ஞ','ண','ந','ம','ன','ய','ர','ல','வ','ழ','ள'];
var totalDataSizeForEachChar = 100;
var totalDataSize = totalDataSizeForEachChar * tamilCharacters.length;

var trainDataSizeForEachChar = (totalDataSizeForEachChar * 80) / 100;
var trainDataSize = tamilCharacters.length * trainDataSizeForEachChar;
var testDataSizeForEachChar = (totalDataSizeForEachChar * 20) / 100;
var testDataSize = tamilCharacters.length * testDataSizeForEachChar;

var trainInputData = new Array(trainDataSize);
var trainInputLabels = new Array(trainDataSize);
var testInputData = new Array(testDataSize);
var testInputLabels = new Array(testDataSize);

async function createInputExample() {
  console.log('Generating Input Examples...');

  // Create a container in the visor
  const surface =
    tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data'});

  var image = new Array(tamilCharacters.length);

  for(var i=0;i<tamilCharacters.length;i++) {
    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = 'margin: 4px; border:1px solid #d3d3d3;';

    image[i]=new Image(28,28);
    image[i].onload = function() {
      var context = canvas.getContext("2d");
      context.drawImage(this, 0,0);
      surface.drawArea.appendChild(canvas);
    }
    var id = getRandomInt(1,totalDataSizeForEachChar);
    image[i].src = "./Images/"+tamilCharacters[i]+"/Preprocessed Images/Ta ("+id.toString()+").jpg";
  }

}

async function createModel() {

  console.log('Creating Model...');

  model = tf.sequential();

  // In the first layer of out convolutional neural network we have
  // to specify the input shape. Then we specify some paramaters for
  // the convolution operation that takes place in this layer.
  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));

  // The MaxPooling layer acts as a sort of downsampling using max values
  // in a region instead of averaging.
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

  // Repeat another conv2d + maxPooling stack.
  // Note that we have more filters in the convolution.
  model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

  // Now we flatten the output from the 2D filters into a 1D vector to prepare
  // it for input into our last layer. This is common practice when feeding
  // higher dimensional data to a final classification output layer.
  model.add(tf.layers.flatten());

  // TESTING PART START
    model.add(tf.layers.dense({
      units: 128,
      activation: 'relu'
    }));
  // TESTING PART END

  // Our last layer (aka output layer) is a dense layer
  const NUM_OUTPUT_CLASSES = tamilCharacters.length;
  model.add(tf.layers.dense({
    units: NUM_OUTPUT_CLASSES,
    kernelInitializer: 'varianceScaling',
    activation: 'softmax'
  }));


  // Choose an optimizer, loss function and accuracy metric,
  // then compile and return the model
  const optimizer = tf.train.adam();
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  const container = {name: 'Model Summary', tab: 'Model'};
  tfvis.show.modelSummary(container, model);
}

async function collectInputs() {
  console.log('Preparing the training data...');

  var image = new Array(totalDataSize);
  var showcaseCanvas = document.getElementById('showcaseCanvas');
  var showcaseContext = showcaseCanvas.getContext('2d');

  for(var i=0;i<tamilCharacters.length;i++) {
    for(var j=0;j<totalDataSizeForEachChar;j++) {
      image[i*totalDataSizeForEachChar +j]=new Image(28,28);
      image[i*totalDataSizeForEachChar +j].src = "./Images/"+tamilCharacters[i]+"/Preprocessed Images/Ta ("+(j+1).toString()+").jpg";
    }
  }

  var count = 0;
  var trainIndex=0;
  var testIndex=0;
  var loop = setInterval(function() {
    showcaseContext.drawImage(image[count],0,0,showcaseCanvas.width, showcaseCanvas.height);
    var canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    var context = canvas.getContext('2d');
    context.drawImage(image[count],0,0);
    var imageData = context.getImageData(0,0,28,28);
    console.log(imageData);

    if(count%5==4) {
      testInputLabels[testIndex] = Math.floor(count/100);
      testInputData[testIndex] = new Array(28*28);
      for (let j = 0; j < imageData.data.length / 4; j++) {
        // All channels hold an equal value since the image is grayscale, so just read the red channel.
        testInputData[testIndex][j] = imageData.data[j * 4] / 255;
      }
      console.log(testInputData[testIndex]);
      testIndex++;
    } else {
      trainInputLabels[trainIndex]=Math.floor(count/100);
      trainInputData[trainIndex] = new Array(28*28);
      for (let j = 0; j < imageData.data.length / 4; j++) {
        // All channels hold an equal value since the image is grayscale, so just read the red channel.
        trainInputData[trainIndex][j] = imageData.data[j * 4] / 255;
      }
      console.log(trainInputData[trainIndex]);
      trainIndex++;
    }
    // console.log(image[count]);
    count++;
    if(count==totalDataSize) {
      clearInterval(loop);
    }
  },10);

}

async function trainModel() {
  console.log("Training the model...");
  //Convert the available data into a suitable format
  //so that it can be converted into tensors
  console.log("Training Data:");
  console.log(trainInputData);
  var batchImagesArray = [].concat.apply([], trainInputData);
  console.log(batchImagesArray);
  var batchLabelsArray = new Array(trainDataSize * tamilCharacters.length).fill(0);
  for(let i=0;i<trainDataSize;i++) {
    batchLabelsArray[i*tamilCharacters.length+trainInputLabels[i]] = 1;
  }
  console.log(batchLabelsArray);

  const [trainXs, trainYs] = tf.tidy(() => {
    const xs = tf.tensor2d(batchImagesArray, [trainDataSize, 28*28]);
    const labels = tf.tensor2d(batchLabelsArray, [trainDataSize, tamilCharacters.length]);
    console.log(xs);
    console.log(labels);
    return [
      xs.reshape([trainDataSize, 28, 28, 1]),
      labels
    ];
  });

  console.log(trainXs);
  console.log(trainYs);

  console.log("Testing/Validation Data:");
  console.log(testInputData);
  var batchImagesArray = [].concat.apply([], testInputData);
  console.log(batchImagesArray);
  var batchLabelsArray = new Array(testDataSize * tamilCharacters.length).fill(0);
  for(let i=0;i<testDataSize;i++) {
    batchLabelsArray[i*tamilCharacters.length+testInputLabels[i]] = 1;
  }
  console.log(batchLabelsArray);

  const [testXs, testYs] = tf.tidy(() => {
    const xs = tf.tensor2d(batchImagesArray, [testDataSize, 28*28]);
    const labels = tf.tensor2d(batchLabelsArray, [testDataSize, tamilCharacters.length]);
    console.log(xs);
    console.log(labels);
    return [
      xs.reshape([testDataSize, 28, 28, 1]),
      labels
    ];
  });

  console.log(testXs);
  console.log(testYs);

  const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
  const container = {
    name: 'Model Training', styles: { height: '1000px' }
  };
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

  return model.fit(trainXs, trainYs, {
    batchSize: 100,
    validationData: [testXs, testYs],
    epochs: 15,
    shuffle: true,
    callbacks: fitCallbacks
  });
}

async function evaluateModel() {
  console.log("Evaluating the model...");
  console.log("Testing/Validation Data:");
  console.log(testInputData);
  var batchImagesArray = [].concat.apply([], testInputData);
  console.log(batchImagesArray);
  var batchLabelsArray = new Array(testDataSize * tamilCharacters.length).fill(0);
  for(let i=0;i<testDataSize;i++) {
    batchLabelsArray[i*tamilCharacters.length+testInputLabels[i]] = 1;
  }
  console.log(batchLabelsArray);

  const [testData, testLabels] = tf.tidy(() => {
    const xs = tf.tensor2d(batchImagesArray, [testDataSize, 28*28]);
    const labels = tf.tensor2d(batchLabelsArray, [testDataSize, tamilCharacters.length]);
    console.log(xs);
    console.log(labels);
    return [
      xs.reshape([testDataSize, 28, 28, 1]),
      labels.argMax([-1])
    ];
  });

  console.log(testData);
  console.log(testLabels);

  const preds = model.predict(testData).argMax([-1]);
  testData.dispose();

  const classAccuracy = await tfvis.metrics.perClassAccuracy(testLabels, preds);
  const accuracyContainer = { name: 'Accuracy', tab: 'Evaluation' };
  tfvis.show.perClassAccuracy(accuracyContainer, classAccuracy, tamilCharacters);

  const confusionMatrix = await tfvis.metrics.confusionMatrix(testLabels, preds);
  const confusionContainer = { name: 'Confusion Matrix', tab: 'Evaluation' };
  tfvis.show.confusionMatrix(confusionContainer, confusionMatrix, tamilCharacters);
}

function getRandomInt(min, max) {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min + 1)) + min;
}


function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            $('#characterImage').attr('src', e.target.result);
            setTimeout(function() {
              identifyImage();
            }, 1000);
        }
        reader.readAsDataURL(input.files[0]);
    }
}

async function identifyImage() {
  var image = document.getElementById('characterImage');
  var canvas = document.createElement('canvas');
  canvas.width = 28;
  canvas.height = 28;
  var context = canvas.getContext('2d');
  context.drawImage(image,0,0);
  var imageData = context.getImageData(0,0,28,28);
  console.log(imageData);

  var requiredImageData = new Array(28*28);
  for (let j = 0; j < imageData.data.length / 4; j++) {
    // All channels hold an equal value since the image is grayscale, so just read the red channel.
    requiredImageData[j] = imageData.data[j * 4] / 255;
  }
  console.log(requiredImageData);

  const [imageDataToIdentify] = tf.tidy(() => {
    const xs = tf.tensor2d(requiredImageData, [1, 28*28]);
    console.log(xs);
    return [
      xs.reshape([1, 28, 28, 1]),
    ];
  });
  console.log(imageDataToIdentify);

  const preds = model.predict(imageDataToIdentify);
  preds.data().then(prediction => console.log(prediction));

  const charIndex = model.predict(imageDataToIdentify).argMax(-1);
  charIndex.data().then(prediction => {
    console.log(prediction);
    var characterImageValue = document.getElementById('characterImageValue');
    characterImageValue.innerHTML = tamilCharacters[prediction[0]];
  });
}

$("#loadImage").change(function(){
    console.log('Loading Character Image...');
    readURL(this);
});

function readWordURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            $('#wordImage').attr('src', e.target.result);
            setTimeout(function() {
              identifyWord();
            }, 1000);
        }
        reader.readAsDataURL(input.files[0]);
    }
}

var wordCharacterCount;

async function identifyWord() {
  var image = document.getElementById('wordImage');
  var canvas = document.createElement('canvas');
  canvas.height = 28;
  canvas.width = (canvas.height * image.width) / image.height;
  var context = canvas.getContext('2d');
  context.drawImage(image,0,0);
  var imageData = context.getImageData(0,0,canvas.width,canvas.height);
  console.log(imageData);

  var imageData2D = new Array(canvas.height);
  for(let j=0;j<canvas.height;j++) {
    imageData2D[j] = new Array(canvas.width);
      for(let i=0;i<canvas.width;i++) {
      imageData2D[j][i]=imageData.data[(canvas.width * 4 * j ) + (i*4)];
    }
  }
  console.log(imageData2D);

  var left=-1, right=-1, top=canvas.height, bottom=-1, prevSpaces=0, currentSpaces=0;
  var imageDimensions=[];
  for(let i=0;i<canvas.width;i++) {
    let scanLineFlag = 0;
    for(let j=0;j<canvas.height;j++) {
      let pixelValue = imageData2D[j][i];
      //Check for some minimum noise threshold
      if(pixelValue < 100) {
        scanLineFlag=1;
        if(top>j) {
          top = j;
        }
        if(bottom<j) {
          bottom = j;
        }
      }
    }
    if(scanLineFlag==1) {
      //Some character has been detected in the vertical scan line
      if(left==-1) {
        left = i;
      } else {
        right = i;
      }
      if(currentSpaces!=0) {
        prevSpaces = currentSpaces;
        currentSpaces = 0;
      }
    } else {
      if(currentSpaces==0) {
        imageDimensions.push([left,right,top,bottom]);
        //Reset for next character
        left=-1;
        top=canvas.height;
        right=-1;
        bottom=-1;
      }
      currentSpaces++;
    }
  }
  if(currentSpaces==0) {
    //Just in case the last character ends on the very last vertical scan line
    imageDimensions.push([left,right,top,bottom]);
  }
  console.log(imageDimensions);

  wordCharacterCount = imageDimensions.length;
  var wordToPrint="";
  var wordImageCharactersDiv = document.getElementById('wordImageCharacters');
  wordImageCharactersDiv.innerHTML="";
  for(let i=0;i<imageDimensions.length;i++) {
    //Context was originally gotten from the word image, directly used here
    let x=imageDimensions[i][0];
    let y=imageDimensions[i][2];
    let width=imageDimensions[i][1]-imageDimensions[i][0];
    let height=imageDimensions[i][3]-imageDimensions[i][2];
    var newImageData = context.getImageData(x,y,width,height);
    let newCanvas = document.createElement('canvas');
    newCanvas.width=width;
    newCanvas.height=height;
    let newContext = newCanvas.getContext('2d');
    newContext.putImageData(newImageData,0,0);

    var imageElement = new Image(28,28);
    imageElement.onload = function() {
      var canvas = document.createElement('canvas');
      canvas.width = 28;
      canvas.height = 28;
      var context = canvas.getContext('2d');
      context.drawImage(this,1,1,26,26);
      var imageData = context.getImageData(0,0,28,28);
      console.log(imageData);

      var requiredImageData = new Array(28*28);
      for (let j = 0; j < imageData.data.length / 4; j++) {
        // All channels hold an equal value since the image is grayscale, so just read the red channel.
        requiredImageData[j] = imageData.data[j * 4] / 255;
      }
      console.log(requiredImageData);

      const [imageDataToIdentify] = tf.tidy(() => {
        const xs = tf.tensor2d(requiredImageData, [1, 28*28]);
        console.log(xs);
        return [
          xs.reshape([1, 28, 28, 1]),
        ];
      });
      console.log(imageDataToIdentify);

      const preds = model.predict(imageDataToIdentify);
      preds.data().then(prediction => console.log(prediction));

      const charIndex = model.predict(imageDataToIdentify).argMax(-1);
      charIndex.data().then(prediction => {
        console.log(prediction);
        this.border=1;
        wordImageCharactersDiv.appendChild(canvas);
        var someText = document.createElement('span');
        someText.style="font-size:50px;";
        someText.innerHTML =tamilCharacters[prediction[0]];
        wordImageCharactersDiv.appendChild(someText);
        wordImageCharactersDiv.appendChild(document.createElement('br'));
        wordToPrint=wordToPrint+tamilCharacters[prediction[0]];
        wordCharacterCount--;
        if(wordCharacterCount==0) {
          document.getElementById('wordImageValue').innerHTML=wordToPrint;
        }
      });
    }
    imageElement.src = newCanvas.toDataURL('image/png');
  }
}

$("#loadWordImage").change(function(){
    console.log('Loading Word Image...');
    readWordURL(this);
});

async function saveTensorFlowModel() {
  const saveResult = await model.save('localstorage://my-model-1');
  console.log('Model saved...');
}

async function loadTensorFlowModel() {
  const modelNew = await tf.loadLayersModel('localstorage://my-model-1');
  console.log('Model loaded successfully...');
  console.log(modelNew);
}
