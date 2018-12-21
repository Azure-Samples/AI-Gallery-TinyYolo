var ndarray = require("ndarray")
var ops = require("ndarray-ops")
var tf = require("@tensorflow/tfjs")
var class_names = require("./yolo_classes")

module.exports = {
  recognizeObjects: function(node) {
    var url = node.src;
    var enc = url.split(",")[1];
  
    if (enc == undefined){
      enc = getImageData(node);
    }
    if (enc == undefined){
      res.innerHTML = 'ERROR: Unable to get source image data';
      return;
    }
    
    var processedFloatArray = preprocess(enc);
  
    var req = new XMLHttpRequest();
    req.responseType = "json";
    req.onreadystatechange = function() {
      if (this.readyState == 4 && this.status == 200) {
        // clear up the result div and add the source picture via canvas
        var result_div = document.getElementById('src');

        // process the response
        var response_body = this.response;
        var inferenceTime = response_body.time;

        postProcess(response_body.result).then(boxes => {
          boxes.forEach(box => {
            const {
              top, left, bottom, right, classProb, className,
            } = box;

            // adjust top/left/bottom/right because we have resized the input picture
            adj_left = Math.round(left * node.naturalWidth / INPUT_DIM);
            adj_top = Math.round(top * node.naturalHeight / INPUT_DIM);
            adj_right = Math.round(right * node.naturalWidth / INPUT_DIM);
            adj_bottom = Math.round(bottom * node.naturalHeight / INPUT_DIM);

            // draw the rectangle for each high prob prediction
            var rect = document.createElement('div');
            rect.className = "yolo_box";
            rect.style.cssText = `top:${adj_top}px; left:${adj_left}px; width:${adj_right-adj_left}px; height:${adj_bottom-adj_top}px;`;

            var label = document.createElement('div');
            label.className = "yolo_label";
            label.innerText = `${className} Confidence: ${Math.round(classProb * 100)}% Time: ${inferenceTime.toFixed(1)}ms`;
            rect.appendChild(label);

            result_div.appendChild(rect);
          });
        });
      }
    };
  
    var url = document.baseURI
    url = url.substring(0, url.lastIndexOf('/'))
    url += '/score'
  
    req.open('POST', url);
    req.setRequestHeader('Content-Type', 'application/json; charset=UTF-8');
    var data = {}
    data["data"] = Array.from(processedFloatArray);
    data["width"] = enc.width;
    data["height"] = enc.height;
    var payload = JSON.stringify(data);
    req.send(payload);
  }
};
const YOLO_ANCHORS = tf.tensor2d([[1.08, 1.19], [3.42, 4.41], 
  [6.63, 11.38], [9.42, 5.11], 
  [16.62, 10.52]
]);

const DEFAULT_FILTER_BOXES_THRESHOLD = 0.01;
const DEFAULT_IOU_THRESHOLD = 0.4;
const DEFAULT_CLASS_PROB_THRESHOLD = 0.3;
const INPUT_DIM = 416;

async function postProcess(result_arrays) {
  outputTensor = tf.transpose(result_arrays, [0, 2, 3, 1]);
  const [boxXy, boxWh, boxConfidence, boxClassProbs ] = yolo_head(outputTensor, YOLO_ANCHORS, class_names.default.length);
  const allBoxes = yolo_boxes_to_corners(boxXy, boxWh);
  const [outputBoxes, scores, classes] = await yolo_filter_boxes(
  allBoxes, boxConfidence, boxClassProbs, DEFAULT_FILTER_BOXES_THRESHOLD);

  // If all boxes have been filtered out
  if (outputBoxes == null) {
    return [];
  }

  const width = tf.scalar(INPUT_DIM);
  const height = tf.scalar(INPUT_DIM);

  const imageDims = tf.stack([height, width, height, width]).reshape([1,4]);

  const boxes = tf.mul(outputBoxes, imageDims);

  const [ preKeepBoxesArr, scoresArr ] = await Promise.all([
    boxes.data(), scores.data(),
  ]);

  const [ keepIndx, boxesArr, keepScores ] = non_max_suppression(
    preKeepBoxesArr,
    scoresArr,
    DEFAULT_IOU_THRESHOLD,
  );

  const classesIndxArr = await classes.gather(tf.tensor1d(keepIndx, 'int32')).data();

  const results = [];

  classesIndxArr.forEach((classIndx, i) => {
    let classProb = keepScores[i];
    if (classProb < DEFAULT_CLASS_PROB_THRESHOLD) {
      return;
    }

    const className = class_names.default[classIndx];
    let [top, left, bottom, right] = boxesArr[i];

    top = Math.max(0, top);
    left = Math.max(0, left);
    bottom = Math.min(INPUT_DIM, bottom);
    right = Math.min(INPUT_DIM, right);

    const resultObj = {
      className,
      classProb,
      bottom,
      top,
      left,
      right,
    };

    results.push(resultObj);
  });
  console.log(results);
  return results;
}

function yolo_filter_boxes(
  boxes,
  boxConfidence,
  boxClassProbs,
  threshold
) {
  const boxScores = tf.mul(boxConfidence, boxClassProbs);
  const boxClasses = tf.argMax(boxScores, -1);
  const boxClassScores = tf.max(boxScores, -1);
  // Many thanks to @jacobgil
  // Source: https://github.com/ModelDepot/tfjs-yolo-tiny/issues/6#issuecomment-387614801
  const predictionMask = tf.greaterEqual(boxClassScores, tf.scalar(threshold)).as1D();

  const N = predictionMask.size;
  // linspace start/stop is inclusive.
  const allIndices = tf.linspace(0, N - 1, N).toInt();
  const negIndices = tf.zeros([N], 'int32');
  const indices = tf.where(predictionMask, allIndices, negIndices);

  return [
    tf.gather(boxes.reshape([N, 4]), indices),
    tf.gather(boxClassScores.flatten(), indices),
    tf.gather(boxClasses.flatten(), indices),
  ];
}

function yolo_boxes_to_corners(box_xy, box_wh) {
  const two = tf.tensor1d([2.0]);
  const box_mins = tf.sub(box_xy, tf.div(box_wh, two));
  const box_maxes = tf.add(box_xy, tf.div(box_wh, two));

  const dim_0 = box_mins.shape[0];
  const dim_1 = box_mins.shape[1];
  const dim_2 = box_mins.shape[2];
  const size = [dim_0, dim_1, dim_2, 1];

  return tf.concat([
    box_mins.slice([0, 0, 0, 1], size),
    box_mins.slice([0, 0, 0, 0], size),
    box_maxes.slice([0, 0, 0, 1], size),
    box_maxes.slice([0, 0, 0, 0], size),
  ], 3);
}

function yolo_head(feats, anchors, num_classes) {
  const num_anchors = anchors.shape[0];

  const anchors_tensor = tf.reshape(anchors, [1, 1, num_anchors, 2]);

  let conv_dims = feats.shape.slice(1, 3);

  // For later use
  const conv_dims_0 = conv_dims[0];
  const conv_dims_1 = conv_dims[1];

  let conv_height_index = tf.range(0, conv_dims[0]);
  let conv_width_index = tf.range(0, conv_dims[1]);
  conv_height_index = tf.tile(conv_height_index, [conv_dims[1]])

  conv_width_index = tf.tile(tf.expandDims(conv_width_index, 0), [conv_dims[0], 1]);
  conv_width_index = tf.transpose(conv_width_index).flatten();

  let conv_index = tf.transpose(tf.stack([conv_height_index, conv_width_index]));
  conv_index = tf.reshape(conv_index, [conv_dims[0], conv_dims[1], 1, 2])
  conv_index = tf.cast(conv_index, feats.dtype);

  feats = tf.reshape(feats, [conv_dims[0], conv_dims[1], num_anchors, num_classes + 5]);
  conv_dims = tf.cast(tf.reshape(tf.tensor1d(conv_dims), [1,1,1,2]), feats.dtype);

  let box_xy = tf.sigmoid(feats.slice([0,0,0,0], [conv_dims_0, conv_dims_1, num_anchors, 2]))
  let box_wh = tf.exp(feats.slice([0,0,0, 2], [conv_dims_0, conv_dims_1, num_anchors, 2]))
  const box_confidence = tf.sigmoid(feats.slice([0,0,0, 4], [conv_dims_0, conv_dims_1, num_anchors, 1]))
  const box_class_probs = tf.softmax(feats.slice([0,0,0, 5],[conv_dims_0, conv_dims_1, num_anchors, num_classes]));

  box_xy = tf.div(tf.add(box_xy, conv_index), conv_dims);
  box_wh = tf.div(tf.mul(box_wh, anchors_tensor), conv_dims);

  return [ box_xy, box_wh, box_confidence, box_class_probs ];
}

function non_max_suppression(boxes, scores, iouThreshold) {
  // Zip together scores, box corners, and index
  const zipped = [];
  for (let i=0; i<scores.length; i++) {
    zipped.push([
      scores[i], [boxes[4*i], boxes[4*i+1], boxes[4*i+2], boxes[4*i+3]], i,
    ]);
  }
  // Sort by descending order of scores (first index of zipped array)
  const sortedBoxes = zipped.sort((a, b) => b[0] - a[0]);

  const selectedBoxes = [];

  // Greedily go through boxes in descending score order and only
  // return boxes that are below the IoU threshold.
  sortedBoxes.forEach((box) => {
    let add = true;
    for (let i=0; i < selectedBoxes.length; i++) {
      // Compare IoU of zipped[1], since that is the box coordinates arr
      // TODO: I think there's a bug in this calculation
      const curIou = box_iou(box[1], selectedBoxes[i][1]);
      if (curIou > iouThreshold) {
        add = false;
        break;
      }
    }
    if (add) {
      selectedBoxes.push(box);
    }
  });

  // Return the kept indices and bounding boxes
  return [
    selectedBoxes.map(e => e[2]),
    selectedBoxes.map(e => e[1]),
    selectedBoxes.map(e => e[0]),
  ];
}

function box_intersection(a, b) {
	const w = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
	const h = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
	if (w < 0 || h < 0) {
		return 0;
	}
	return w * h;
}

function box_union(a, b) {
	const i = box_intersection(a, b);
	return (a[3] - a[1]) * (a[2] - a[0]) + (b[3] - b[1]) * (b[2] - b[0]) - i;
}

function box_iou(a, b) {
	return box_intersection(a, b) / box_union(a, b);
}

function getImageData(imgElem) {
  // imgElem must be on the same server otherwise a cross-origin error will be thrown "SECURITY_ERR: DOM Exception 18"
  var canvas = document.createElement("canvas");
  canvas.width = INPUT_DIM;
  canvas.height = INPUT_DIM;
  var ctx = canvas.getContext("2d");
  ctx.drawImage(imgElem, 0, 0, imgElem.naturalWidth, imgElem.naturalHeight, 0, 0, canvas.width, canvas.height);
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

function preprocess(imageData) {
  const { data, width, height } = imageData;
  // data processing
  const dataTensor = ndarray(new Float32Array(data), [width, height, 4]);
  const dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [1, 3, width, height]);

  ops.assign(dataProcessedTensor.pick(0, 0, null, null), dataTensor.pick(null, null, 2));
  ops.assign(dataProcessedTensor.pick(0, 1, null, null), dataTensor.pick(null, null, 1));
  ops.assign(dataProcessedTensor.pick(0, 2, null, null), dataTensor.pick(null, null, 0));
  
  return dataProcessedTensor.data;
}
