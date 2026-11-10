/**
 * Windy-style Dynamic Particle Wind Stream Engine
 * High-performance HTML5 Canvas 2D Vector Streamline Animation.
 */
(function (global) {
  "use strict";

  function clamp(val, min, max) {
    return Math.max(min, Math.min(max, val));
  }

  // Wind speed color scale: Beaufort scale gradient
  function getWindColor(speed) {
    if (speed < 6.0) return "rgba(56, 189, 248, 0.75)";   // Gentle: Light Cyan
    if (speed < 12.0) return "rgba(74, 222, 128, 0.85)";  // Moderate: Emerald Green
    if (speed < 18.0) return "rgba(250, 204, 21, 0.90)";  // Fresh: Vivid Yellow
    if (speed < 24.0) return "rgba(251, 146, 60, 0.95)";  // Strong/Gale: Bright Orange
    return "rgba(239, 68, 68, 0.98)";                     // Storm/Typhoon: Crimson Red
  }

  class WindParticleEngine {
    constructor(canvas, options = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.options = Object.assign({
        particleCount: 1000,
        speedFactor: 0.35,
        maxAge: 70,
        trailOpacity: 0.92,
        lineWidth: 1.8,
        clearMode: "transparent", // "transparent" for map overlay, "solid" for dark canvas
      }, options);

      this.gridData = null;
      this.particles = [];
      this.animId = null;
      this.isRunning = false;
      this.projectFn = null;  // (lon, lat) => [px, py]
      this.unprojectFn = null;// (px, py) => [lon, lat]

      this._resize();
    }

    setProjections(project, unproject) {
      this.projectFn = project;
      this.unprojectFn = unproject;
    }

    setParticleCount(count) {
      this.options.particleCount = count;
      this._initParticles();
    }

    clear() {
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      this._initParticles();
    }

    _resize() {
      const dpr = window.devicePixelRatio || 1;
      const rect = this.canvas.getBoundingClientRect();
      const w = Math.round(rect.width || this.canvas.clientWidth || 800);
      const h = Math.round(rect.height || this.canvas.clientHeight || 500);

      if (this.canvas.width !== w * dpr || this.canvas.height !== h * dpr) {
        this.canvas.width = w * dpr;
        this.canvas.height = h * dpr;
        this.ctx.scale(dpr, dpr);
      }
      this.width = w;
      this.height = h;
    }

    setData(gridData) {
      this.gridData = gridData;
      this._buildLookup();
      this._initParticles();
    }

    _buildLookup() {
      if (!this.gridData || !this.gridData.cells) return;
      const { bbox, grid_dimensions } = this.gridData;
      this.west = bbox[0];
      this.south = bbox[1];
      this.east = bbox[2];
      this.north = bbox[3];
      this.cols = grid_dimensions.cols;
      this.rows = grid_dimensions.rows;

      this.uGrid = new Float32Array(this.cols * this.rows);
      this.vGrid = new Float32Array(this.cols * this.rows);
      this.speedGrid = new Float32Array(this.cols * this.rows);
      this.confGrid = new Float32Array(this.cols * this.rows);

      for (const cell of this.gridData.cells) {
        const c = cell.col;
        const r = cell.row;
        if (c >= 0 && c < this.cols && r >= 0 && r < this.rows) {
          const idx = r * this.cols + c;
          this.uGrid[idx] = cell.u_component_mps || 0.0;
          this.vGrid[idx] = cell.v_component_mps || 0.0;
          this.speedGrid[idx] = cell.wind_speed_mps || 0.0;
          this.confGrid[idx] = cell.spatial_confidence ?? cell.confidence_weight ?? 0.0;
        }
      }
    }

    // Bilinear interpolation of wind vector at (lon, lat)
    getWindVector(lon, lat) {
      if (!this.gridData || !this.uGrid) return null;
      if (lon < this.west || lon > this.east || lat < this.south || lat > this.north) {
        return null;
      }

      const fx = ((lon - this.west) / (this.east - this.west)) * (this.cols - 1);
      const fy = ((lat - this.south) / (this.north - this.south)) * (this.rows - 1);

      const x0 = clamp(Math.floor(fx), 0, this.cols - 1);
      const x1 = clamp(x0 + 1, 0, this.cols - 1);
      const y0 = clamp(Math.floor(fy), 0, this.rows - 1);
      const y1 = clamp(y0 + 1, 0, this.rows - 1);

      const sx = fx - x0;
      const sy = fy - y0;

      const idx00 = y0 * this.cols + x0;
      const idx10 = y0 * this.cols + x1;
      const idx01 = y1 * this.cols + x0;
      const idx11 = y1 * this.cols + x1;

      const interp = (grid) => {
        const top = (1.0 - sx) * grid[idx00] + sx * grid[idx10];
        const bottom = (1.0 - sx) * grid[idx01] + sx * grid[idx11];
        return (1.0 - sy) * top + sy * bottom;
      };

      const u = interp(this.uGrid);
      const v = interp(this.vGrid);
      const confidence = this.confGrid ? interp(this.confGrid) : 1.0;
      const speed = Math.hypot(u, v);
      const direction = (Math.atan2(-u, -v) * (180.0 / Math.PI) + 360.0) % 360.0;

      return { u, v, speed, direction, confidence };
    }

    _initParticles() {
      this.particles = [];
      const count = this.options.particleCount;
      for (let i = 0; i < count; i++) {
        this.particles.push(this._createParticle(true));
      }
    }

    _createParticle(randomAge = false) {
      const x = Math.random() * this.width;
      const y = Math.random() * this.height;
      const maxAge = Math.floor(35 + Math.random() * this.options.maxAge);
      const age = randomAge ? Math.floor(Math.random() * maxAge) : 0;
      return { x, y, age, maxAge };
    }

    start() {
      if (this.isRunning) return;
      this.isRunning = true;
      this._loop();
    }

    stop() {
      this.isRunning = false;
      if (this.animId) {
        cancelAnimationFrame(this.animId);
        this.animId = null;
      }
    }

    _loop() {
      if (!this.isRunning) return;
      this._renderFrame();
      this.animId = requestAnimationFrame(() => this._loop());
    }

    _renderFrame() {
      this._resize();
      const ctx = this.ctx;
      const w = this.width;
      const h = this.height;

      if (this.options.clearMode === "solid") {
        ctx.fillStyle = `rgba(7, 13, 23, ${1.0 - (this.options.trailOpacity || 0.92)})`;
        ctx.fillRect(0, 0, w, h);
      } else {
        // Transparent trail for overlaying Leaflet maps
        ctx.globalCompositeOperation = "destination-out";
        ctx.fillStyle = `rgba(0, 0, 0, ${1.0 - (this.options.trailOpacity || 0.92)})`;
        ctx.fillRect(0, 0, w, h);
        ctx.globalCompositeOperation = "source-over";
      }

      if (!this.gridData || !this.unprojectFn || !this.projectFn) {
        return;
      }

      ctx.lineWidth = this.options.lineWidth;
      ctx.lineCap = "round";

      for (let i = 0; i < this.particles.length; i++) {
        const p = this.particles[i];
        if (p.age >= p.maxAge) {
          this.particles[i] = this._createParticle(false);
          continue;
        }

        const geo = this.unprojectFn(p.x, p.y);
        if (!geo) {
          this.particles[i] = this._createParticle(false);
          continue;
        }

        const wind = this.getWindVector(geo[0], geo[1]);
        if (!wind || wind.speed < 0.2) {
          p.age += 2;
          continue;
        }

        // Particle displacement along wind vector
        // Note: positive u moves East (right), positive v moves North (up in geo, down in pixel y)
        const dx = wind.u * this.options.speedFactor;
        const dy = -wind.v * this.options.speedFactor;

        const nextX = p.x + dx;
        const nextY = p.y + dy;

        if (nextX < 0 || nextX > w || nextY < 0 || nextY > h) {
          this.particles[i] = this._createParticle(false);
          continue;
        }

        ctx.strokeStyle = getWindColor(wind.speed);
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(nextX, nextY);
        ctx.stroke();

        p.x = nextX;
        p.y = nextY;
        p.age++;
      }
    }
  }

  global.WindParticleEngine = WindParticleEngine;
})(window);
