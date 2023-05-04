/**
 * Fast KMEANS clustering based on multi-granularity
 * @constructor
 *
 * @param {Array} dataset
 * @param {number} k - number of clusters
 * @param {function} distance - distance function
 * @returns {KMEANS}
 */
let maxDim = 10;
export function FastKMEANS(dataset, k, distance) {
  // number of clusters
  this.k = k || 3;
  this.dataset = dataset || []; // set of feature vectors

  //每个样本点所属的聚类簇的标签
  this.assignments = [];

  // 聚类中心
  this.centroids = [];

  // 到最近聚类中心的距离
  this.upperBounds = [];

  // 每个数据点到其次近聚类中心的距离
  this.lowerBounds = [];

  // neighbor clusters of each cluster
  this.neighborClusters = [];

  this.init(distance);
}

/**
 * @returns {undefined}
 */
FastKMEANS.prototype.init = function (distance) {
  this.assignments = [];
  this.centroids = [];
  this.upperBounds = [];
  this.lowerBounds = [];
  this.neighborClusters = [];

  if (typeof distance !== 'undefined') {
    this.distance = distance;
  }
};

/**
 * @returns {undefined}
 */
FastKMEANS.prototype.run = function () {
  this.init();
  let len = this.dataset.length;

  // initialize centroids
  for (let i = 0; i < this.k; i++) {
    this.centroids[i] = this.randomCentroid();
  }

  // 基于两个最近的质心初始化上限和下限
  for (let i = 0; i < len; i++) {
    let closestFirstCentroid = this.argmin(this.dataset[i], this.centroids, this.distance);
    let minDistance = Number.MAX_VALUE;

    // 次近的质心
    let secondClosestCentroid;

    for (let centroidId = 0; centroidId < this.k; centroidId++) {
      if (centroidId !== closestFirstCentroid) {
        let dist = this.distance(this.dataset[i], this.centroids[centroidId]);
        if (dist < minDistance) {
          minDistance = dist;
          secondClosestCentroid = centroidId;
        }
      }
    }
    this.upperBounds[i] = minDistance;

    //次近
    this.lowerBounds[i] = this.distance(this.dataset[i], this.centroids[secondClosestCentroid]);
    this.assignments[i] = closestFirstCentroid;
  }

  // 更新边界和相邻簇直到收敛，迭代次数T
  let change = true;
  while (change) {
    change = false;
    this.updateNeighborClusters();

    // 调整质心的位置
    for (let centroidId = 0; centroidId < this.k; centroidId++) {
      // 计算聚类中心，dim是每个向量的维数
      let mean = new Array(maxDim);
      let count = 0;

      // init mean vector
      for (let dim = 0; dim < maxDim; dim++) {
        mean[dim] = 0;
      }

      //优化：
      for (let j = 0; j < len; j++) {
        let maxDim = this.dataset[j].length;

        // if current cluster id is assigned to point
        if (centroidId === this.assignments[j]) {
          for (let dim = 0; dim < maxDim; dim++) {
            mean[dim] += this.dataset[j][dim];
          }
          count++;
        }
      }

      if (count > 0) {
        // 如果簇包含点，则调整质心位置
        for (let dim = 0; dim < maxDim; dim++) {
          mean[dim] /= count;
        }
        //


        // 检查质心是否已移动 //todo
        if (!arraysEqual(mean, this.centroids[centroidId])) {
          this.centroids[centroidId] = mean;
          change = true;
        }
      } else {
        // 如果簇为空，则生成新的随机质心
        this.centroids[centroidId] = this.randomCentroid();
        change = true;
      }
    }

    // 更新上限和下限
    for (let i = 0; i < len; i++) {
      let oldUpperBound = this.upperBounds[i];
      let oldLowerBound = this.lowerBounds[i];
      let distToClosestCentroid = this.distance(this.dataset[i], this.centroids[this.assignments[i]]);

      if (distToClosestCentroid < oldUpperBound) {
        this.upperBounds[i] = distToClosestCentroid;

        let neighborClusters = this.neighborClusters[this.assignments[i]];

        // max todo
        for (let nc = 0; nc < neighborClusters?.length; nc++) {
          let neighborClusterId = neighborClusters[nc];
          if (neighborClusterId !== this.assignments[i]) {
            let distToNeighborCentroid = this.distance(this.dataset[i], this.centroids[neighborClusterId]);
            if (distToNeighborCentroid < oldLowerBound) {
              this.lowerBounds[i] = distToNeighborCentroid;
              this.assignments[i] = neighborClusterId;
              change = true;
              break;
            }
          }
        }
      }
    }
  }

  return this.getClusters();
};

/**
 * Generate random centroid
 *
 * @returns {Array}
 */
FastKMEANS.prototype.randomCentroid = function () {
  let maxId = this.dataset.length - 1;
  let centroid;
  let id;

  do {
    id = Math.round(Math.random() * maxId);
    centroid = this.dataset[id];
  } while (this.centroids.indexOf(centroid) >= 0);

  return centroid;
};

/**
 * 计算指定聚类簇的邻居簇和邻居簇之间的 lowerBounds 和 upperBounds
 * Assign points to clusters
 *
 * @returns {boolean}
 */
FastKMEANS.prototype.updateNeighborClusters = function () {
  if (this.assignments === undefined) {
    return; // or throw an error
  }
  let len = this.dataset.length;
  let neighborClusters = [];

  for (let i = 0; i < this.k; i++) {
    let neighborClusterIds = new Set();

    for (let j = 0; j < len; j++) {
      if (this.assignments[j] === i) {
        let neighborClustersOfPoint = this.neighborClusters[j];
        if (!neighborClustersOfPoint) return;
        for (let ncof = 0; ncof < neighborClustersOfPoint.length; ncof++) {
          neighborClusterIds.add(neighborClustersOfPoint[ncof]);
        }
      }
    }

    neighborClusterIds.delete(i);
    neighborClusters[i] = Array.from(neighborClusterIds);
  }

  this.neighborClusters = neighborClusters;
};

/**
 * Extract information about clusters
 *
 * @returns {undefined}
 */
FastKMEANS.prototype.getClusters = function () {
  let clusters = new Array(this.k);
  let centroidId;

  for (let pointId = 0; pointId < this.assignments.length; pointId++) {
    centroidId = this.assignments[pointId];

    // init empty cluster
    if (typeof clusters[centroidId] === 'undefined') {
      clusters[centroidId] = [];
    }

    clusters[centroidId].push(pointId);
  }

  return clusters;
};

// utils

/**
 * @params 找到距离当前样本点最近的聚类中心点
 * {Array} point
 * @params {Array.<Array>} set
 * @params {Function} f
 * @returns {number}
 */
FastKMEANS.prototype.argmin = function (point, set, f) {
  let min = Number.MAX_VALUE;
  let arg = 0;
  let len = set.length;
  let d;

  for (let i = 0; i < len; i++) {
    d = f(point, set[i]);
    if (d < min) {
      min = d;
      arg = i;
    }
  }

  return arg;
};

/**
 * Euclidean distance
 *
 * @params {number} p
 * @params {number} q
 * @returns {number}
 */
FastKMEANS.prototype.distance = function (p, q) {
  let sum = 0;
  let i = Math.min(p.length, q.length);

  while (i--) {
    let diff = p[i] - q[i];
    sum += diff * diff;
  }

  return Math.sqrt(sum);
};

/**
 * Helper function to check if two arrays are equal
 * @param {Array} a
 * @param {Array} b
 * @returns {boolean}
 */
function arraysEqual(a, b) {
  if (a === b) {
    return true;
  }
  if (a == null || b == null) {
    return false;
  }
  if (a.length !== b.length) {
    return false;
  }

  for (let i = 0; i < a.length; ++i) {
    if (a[i] !== b[i]) {
      return false;
    }
  }

  return true;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = FastKMEANS;
}
