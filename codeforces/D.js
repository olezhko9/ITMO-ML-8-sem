var str = readline().split(' ');
var N = parseInt(str[0]), M = parseInt(str[1])

start = +new Date()
data = [];
for (var i = 0; i < N; i++) {
  data[i] = readline().split(' ').map(x => parseInt(x, 10));
}

features_mean = [];
features_std = [];

zip = matrix => {
  return matrix[0].map((_, i) => {
    return matrix.map((array) => array[i]);
  });
};

sum = arr => {
  var res = 0;
  for (var i = 0; i < arr.length; i++) {
    res += arr[i];
  }
  return res;
};

for (var feature_values of zip(data)) {
  var mean = sum(feature_values) / N;
  features_mean.push(mean);
  var std = 0.0;
  for (var i = 0; i < feature_values.length; i++) {
    std += Math.pow(feature_values[i] - mean, 2);
  }
  std = Math.pow(std / (N - 1), 0.5);
  features_std.push(std);
}


for (var i = 0; i < data.length; i++) {
  for (var j = 0; j < M + 1; j++) {
    if (features_std[j] === 0.0) {
      data[i][j] = 0.0
    } else {
      data[i][j] = (data[i][j] - features_mean[j]) / features_std[j];
    }
  }
}

var lr = 0.0005;

var weights = [];
for (var i = 0; i < M + 1; i++) {
  weights[i] = Math.random();
}

var randint = (min, max) => {
  return Math.floor(Math.random() * (max - min)) + min;
};

while ((+new Date() - start) < 930) {
  var train = data[randint(0, N)];
  var y_pred = weights.slice(-1)[0];
  for (var i = 0; i < M; i++) {
    y_pred += weights[i] * train[i];
  }
  var error = y_pred - train.slice(-1)[0];
  for (var i = 0; i < M; i++) {
    weights[i] -= lr * error * train[i];
  }
  weights[weights.length - 1] -= lr * error;
}

var bias = weights.slice(-1)[0];
for (var i = 0; i < M; i++) {
  if (features_std[i] === 0.0) {
    print(weights[i])
    continue;
  }
  var t = weights[i] * features_std.slice(-1)[0] / features_std[i];
  bias -= t * features_mean[i];
  print(t, '\n');
}
print(bias + features_mean.slice(-1)[0], '\n');
