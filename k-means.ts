
class KMEANS {
  k: number; // number of clusters
  dataset: number[][]; // set of feature vectors
  assignments: number[]; // set of associated clusters for each feature vector
  centroids: number[][]; // vectors for our clusters
  distance: (p: number[], q: number[]) => number; // distance function

  constructor(dataset: number[][], k: number, distance: (p: number[], q: number[]) => number) {
    this.k = k ?? 3;
    this.dataset = dataset ?? [];
    this.assignments = [];
    this.centroids = [];
    this.distance = distance ?? ((p, q) => this.euclideanDistance(p, q));
  }

  /**
   * @returns {undefined}
   */
  init(dataset?: number[][], k?: number, distance?: (p: number[], q: number[]) => number): void {
    this.assignments = [];
    this.centroids = [];

    if (dataset) {
      this.dataset = dataset;
    }

    if (k) {
      this.k = k;
    }

    if (distance) {
      this.distance = distance;
    }
  }

  /**
   * @returns {undefined}
   */
  run(dataset?: number[][], k?: number): number[][][] {
    this.init(dataset, k);

    const len = this.dataset.length;

    // initialize centroids
    for (let i = 0; i < this.k; i++) {
      this.centroids[i] = this.randomCentroid();
    }

    let change = true;
    while (change) {

      // assign feature vectors to clusters
      change = this.assign();

      // adjust location of centroids
      for (let centroidId = 0; centroidId < this.k; centroidId++) {
        let mean = new Array(this.dataset[0].length).fill(0);
        let count = 0;

        for (let j = 0; j < len; j++) {
          // if current cluster id is assigned to point
          if (centroidId === this.assignments[j]) {
            for (let dim = 0; dim < mean.length; dim++) {
              mean[dim] += this.dataset[j][dim];
            }
            count++;
          }
        }

        if (count > 0) {
          // if cluster contain points, adjust centroid position
          for (let dim = 0; dim < mean.length; dim++) {
            mean[dim] /= count;
          }

          this.centroids[centroidId] = mean;
        } else {
          // if cluster is empty, generate new random centroid
          this.centroids[centroidId] = this.randomCentroid();
          change = true;
        }
      }
    }

    return this.getClusters();
  }

  /**
   * Generate random centroid
   *
   * @returns {Array}
   */
  randomCentroid(): number[] {
    const maxId = this.dataset.length - 1;
    let centroid;
    let id;

    do {
      id = Math.round(Math.random() * maxId);
      centroid = this.dataset[id];
    } while (this.centroids.indexOf(centroid) >= 0);

    return centroid;
  }

  /**
   * Assign points to clusters
   *
   * @returns {boolean}
   */
  assign(): boolean {
    let change = false;
    const len = this.dataset.length;
    let closestCentroid: number;

    for (let i = 0; i < len; i++) {
      closestCentroid = this.argmin(this.dataset[i], this.centroids, this.distance);

      if (closestCentroid !== this.assignments[i]) {
        this.assignments[i] = closestCentroid;
        change = true;
      }
    }

    return change;
  }

  /**
   * Extract information about clusters
   *
   * @returns {undefined}
   */
  getClusters(): number[][][] {
    const clusters = new Array(this.k);
    let centroidId: number;

    for (let pointId = 0; pointId < this.assignments.length; pointId++) {
      centroidId = this.assignments[pointId];

      // init empty cluster
      if (typeof clusters[centroidId] === 'undefined') {
        clusters[centroidId] = [];
      }

      clusters[centroidId].push(pointId);
    }

    return clusters.map(ids => ids.map(id => this.dataset[id]));
  }

  // utils

  /**
   * @params {Array} point
   * @params {Array.<Array>} set
   * @params {Function} f
   * @returns {number}
   */
  argmin(point: number[], set: number[][], f: (p: number[], q: number[]) => number): number {
    let min = Number.MAX_VALUE;
    let arg = 0;
    const len = set.length;
    let d;

    for (let i = 0; i < len; i++) {
      d = f(point, set[i]);
      if (d < min) {
        min = d;
        arg = i;
      }
    }

    return arg;
  }

  /**
   * Euclidean distance
   *
   * @params {number} p
   * @params {number} q
   * @returns {number}
   */
  euclideanDistance(p: number[], q: number[]): number {
    let sum = 0;
    const dim = Math.min(p.length, q.length);

    for (let i = 0; i < dim; i++) {
      const diff = p[i] - q[i];
      sum += diff * diff;
    }

    return Math.sqrt(sum);
  }
}

export default KMEANS;
