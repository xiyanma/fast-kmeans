/**
 * Fast KMEANS clustering based on multi-granularity
 * @constructor
 *
 * @param {Array} dataset
 * @param {number} k - number of clusters
 * @param {function} distance - distance function
 * @returns {KMEANS}
 */
const maxDim = 10;
export class FastKMEANS {
  // 聚类数量
  private k: number;
  private dataset: Array<Array<number>>;

  //每个样本点所属的聚类簇的标签
  private assignments: Array<number>;

  // 聚类中心
  private centroids: Array<Array<number>>;

  // 到最近聚类中心的距离
  private upperBounds: Array<number>;

  // 每个数据点到其次近聚类中心的距离
  private lowerBounds: Array<number>;

  // 每个簇的相邻簇
  private neighborClusters: Array<Array<number>>;
  private distance!: Function;

  constructor(dataset: Array<Array<number>>, k: number, distance?: Function) {
    this.k = k || 3;
    this.dataset = dataset || [];
    this.assignments = [];
    this.centroids = [];
    this.upperBounds = [];
    this.lowerBounds = [];
    this.neighborClusters = [];

    this.init(distance);
  }

  init(distance?: Function | undefined) {
    this.assignments = [];
    this.centroids = [];
    this.upperBounds = [];
    this.lowerBounds = [];
    this.neighborClusters = [];
    this.distance = distance || euclideanDistance;
  }

  run() {
    this.init();
    let len = this.dataset.length;

    // 初始化质心
    for (let i = 0; i < this.k; i++) {
      this.centroids[i] = this.randomCentroid();
    }

    // 基于两个最近的质心初始化上限和下限
    for (let i = 0; i < len; i++) {
      const closestFirstCentroid = argmin(this.dataset[i], this.centroids, this.distance);
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

      // 次近
      if (secondClosestCentroid) {
        this.lowerBounds[i] = this.distance(this.dataset[i], this.centroids[secondClosestCentroid]);
      }
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

        // 初始化 mean vector
        for (let dim = 0; dim < maxDim; dim++) {
          mean[dim] = 0;
        }

        for (let j = 0; j < len; j++) {
          let maxDim = this.dataset[j].length;

          // 如果当前簇id已分配给点
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

          // 检查质心是否已移动
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
  }

  /**
   *生成随机质心
   *
   * @returns {Array}
   */
  randomCentroid() {
    const maxId = this.dataset.length - 1;
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
   * 为簇指定点
   * @returns {boolean}
   */
  updateNeighborClusters() {
    if (this.assignments === undefined) {
      return;
    }
    let len = this.dataset.length;
    let neighborClusters: Array<Array<number>> = [];

    for (let i = 0; i < this.k; i++) {
      let neighborClusterIds = new Set<number>();

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
   * 返回簇的集合
   *
   * @returns {undefined}
   */
  getClusters() {
    let clusters = new Array(this.k);
    let centroidId;

    for (let pointId = 0; pointId < this.assignments.length; pointId++) {
      centroidId = this.assignments[pointId];

      // 初始化空簇
      if (typeof clusters[centroidId] === 'undefined') {
        clusters[centroidId] = [];
      }

      clusters[centroidId].push(pointId);
    }

    return clusters;
  };
}

/**
 * 欧几里得距离
 *
 * @params {number} p
 * @params {number} q
 * @returns {number}
 */
const euclideanDistance = (p: Array<number>, q: Array<number>) => {
  let sum = 0;
  let i = Math.min(p.length, q.length);

  while (i--) {
    let diff = p[i] - q[i];
    sum += diff * diff;
  }

  return Math.sqrt(sum);
};

/**
 * @params 找到距离当前样本点最近的聚类中心点
 * {Array} point
 * @params {Array.<Array>} set
 * @params {Function} f
 * @returns {number}
 */
const argmin = (point: any, set: string | any[], distance: Function) => {
  const len = set.length;
  let min = Number.MAX_VALUE;
  let arg = 0;
  let d;

  for (let i = 0; i < len; i++) {
    d = distance(point, set[i]);
    if (d < min) {
      min = d;
      arg = i;
    }
  }
  return arg;
};

/**
 * 检查两个数组是否相等
 * @param {Array} a
 * @param {Array} b
 * @returns {boolean}
 */
const arraysEqual = (a: Array<number>, b: Array<number>) => {
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
};
