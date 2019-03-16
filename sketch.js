function setup() {
  canvas = createCanvas(200, 200);
  canvas.parent('characterCanvas');
  background(255);
}

function draw() {
  stroke(0);
  if (mouseIsPressed === true) {
    line(mouseX, mouseY, pmouseX, pmouseY);
    strokeWeight(10);
  }
}

function characterCanvasClear() {
  background(255);
}

function identifyCharacterCanvas() {
  console.log("Identifying the canvas character...");
  var originalCanvas = document.getElementById('defaultCanvas0');
  // var originalContext = originalCanvas.getContext('2d');
  // var originalImageData = originalContext.getImageData(0,0,200,200);

  var image = new Image(28,28);
  image.onload = function() {
      var canvas = document.createElement('canvas');
      canvas.width = 28;
      canvas.height = 28;
      var context = canvas.getContext('2d');
      context.drawImage(this,0,0,28,28);
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
        var characterImageValue = document.getElementById('characterCanvasValue');
        characterImageValue.innerHTML = tamilCharacters[prediction[0]];
      });

      characterCanvasClear();
  }
  image.src = originalCanvas.toDataURL('image/png');
}
