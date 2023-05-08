/**
 * Fast KMEANS clustering based on multi-granularity
 * @constructor
 *
 * @param {Array} dataset
 * @param {number} k - number of clusters
 * @param {function} distance - distance function
 * @returns {KMEANS}
 */

type Point = number[];

export class FastKMEANS {
  // 聚类数量
  private k: number;
  private dataset: Point[];

  // 每个样本点所属的聚类簇的id
  private assignments: Array<number>;

  // 聚类中心
  private centroids: Point[];

  // 到最近聚类中心的距离
  private upperBounds: Array<number>;

  // 每个数据点到其次近聚类中心的距离
  private lowerBounds: Array<number>;

  // 每个簇的相邻簇
  private neighborClusters: Array<Array<number>>;
  private distance: (a: Point, b: Point) => number;

  constructor(dataset: Point[], k: number = 3, distance: (a: Point, b: Point) => number = euclideanDistance) {
    this.k = k;
    this.dataset = dataset;
    this.neighborClusters = new Array(this.dataset.length).fill(null).map(() => []);
    this.distance = distance;
    const len = this.dataset.length;
    this.assignments = new Array(len).fill(-1);
    this.centroids = new Array(this.k);
    this.upperBounds = new Array(len).fill(undefined);
    this.lowerBounds = new Array(len).fill(undefined);
    for (let i = 0; i < this.k; i++) {
      this.centroids[i] = this.randomCentroid();
    }
  }

  public run() {
    const len = this.dataset.length;

    // 初始化质心。基于两个最近的质心初始化上限和下限
    for (let i = 0; i < len; i++) {
      const closestFirstCentroid = argmin(this.dataset[i], this.centroids, this.distance);
      let minDistance = Number.MAX_SAFE_INTEGER;

      // 次近的质心
      let secondClosestCentroid = -1;
      for (let centroidId = 0; centroidId < this.k; centroidId++) {
        if (centroidId !== closestFirstCentroid) {
          const dist = this.distance(this.dataset[i], this.centroids[centroidId]);
          if (dist < minDistance) {
            minDistance = dist;
            secondClosestCentroid = centroidId;
          }
        }
      }
      this.upperBounds[i] = minDistance;
      if (secondClosestCentroid !== -1) {
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
        // 计算聚类中心，dim是每个点的维数
        const mean = new Array(this.dataset[0].length).fill(0);
        let count = 0;
        for (let j = 0; j < len; j++) {
          // 如果当前簇id已分配给点
          if (centroidId === this.assignments[j]) {
            for (let dim = 0; dim < mean.length; dim++) {
              mean[dim] += this.dataset[j][dim];
            }
            count++;
          }
        }
        if (count > 0) {
          // 如果簇包含点，则调整质心位置
          for (let dim = 0; dim < mean.length; dim++) {
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
        const oldUpperBound = this.upperBounds[i];
        const oldLowerBound = this.lowerBounds[i];
        const distToClosestCentroid = this.distance(this.dataset[i], this.centroids[this.assignments[i]]);
        if (distToClosestCentroid < oldUpperBound) {
          this.upperBounds[i] = distToClosestCentroid;
          const neighbors = this.neighborClusters[this.assignments[i]];
          for (let nc = 0; nc < neighbors.length; nc++) {
            const neighborClusterId = neighbors[nc];
            if (neighborClusterId !== this.assignments[i]) {
              const distToNeighborCentroid = this.distance(this.dataset[i], this.centroids[neighborClusterId]);
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
   * 生成随机质心
   *
   * @returns {Point}
   */
  private randomCentroid() {
    let centroid: Point;
    const usedIds = new Set();
    do {
      const id = Math.round(Math.random() * (this.dataset.length - 1));
      centroid = this.dataset[id];
      usedIds.add(id);
    } while (usedIds.size < this.k && usedIds.size < this.dataset.length && this.centroids.includes(centroid));
    return centroid;
  };

  /**
   * 计算指定聚类簇的邻居簇和邻居簇之间的 lowerBounds 和 upperBounds
   */
  private updateNeighborClusters() {
    const len = this.dataset.length;
    const neighborClusters: Array<Array<number>> = [];

    for (let i = 0; i < this.k; i++) {
      const neighborClusterIds = new Set<number>();
      for (let j = 0; j < len; j++) {
        if (this.assignments[j] === i) {
          const neighborClustersOfPoint = this.neighborClusters[j];
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
   * @returns {Array}
   */
  private getClusters() {
    const clusters = new Map();
    for (let i = 0; i < this.dataset.length; i++) {
      const clusterId = this.assignments[i];
      if (!clusters.has(clusterId)) {
        // 初始化空簇
        clusters.set(clusterId, []);
      }
      clusters.get(clusterId).push(i);
    }
    return Array.from(clusters.values());
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
const argmin = (point: Array<number>, set: Array<Array<number>>, distance: Function) => {
  const len = set.length;
  let min = Number.MAX_SAFE_INTEGER;
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
  if (!a || !b) {
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
