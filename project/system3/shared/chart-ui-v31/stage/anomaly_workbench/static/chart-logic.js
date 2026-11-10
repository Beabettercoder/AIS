(function (root, factory) {
  'use strict';
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.AISChartLogic = factory();
}(typeof window !== 'undefined' ? window : this, function () {
  'use strict';
  const valid = value => value !== null && value !== undefined && value !== '' && typeof value !== 'boolean' && Number.isFinite(Number(value));
  const kinds = [
    ['turn_back', '明显折返'], ['deviation', '方向变化线索'], ['stall', '快速减速后持续低速'],
    ['sinuous', '持续大幅交替转向'], ['speed_drop', '持续快速减速'],
    ['formation_departure', '疑似编队脱离'], ['group_inconsistency', '疑似群体运动不一致'],
    ['congestion', '区域拥堵迹象'], ['observation_glitch', '观测质量疑点']
  ];

  function timePositions(times, count) {
    const size = Math.max(0, Math.floor(Number(count) || 0));
    const values = Array.from({length: size}, (_, i) => valid(times?.[i]) ? Number(times[i]) : null);
    const known = values.filter(value => value !== null);
    if (!known.length) return values;
    const first = Math.min(...known), span = Math.max(...known) - first;
    return values.map(value => value === null ? null : span > 0 ? (value - first) / span : 0);
  }

  function seriesSegments(values, {circular = false} = {}) {
    const result = [];
    let segment = [];
    function finish() { if (segment.length) result.push(segment); segment = []; }
    (Array.isArray(values) ? values : []).forEach((value, i) => {
      if (!valid(value)) { finish(); return; }
      if (circular && segment.length && Math.abs(Number(value) - Number(values[segment[segment.length - 1]])) > 180) finish();
      segment.push(i);
    });
    finish();
    return result;
  }

  function originalSeries(record) {
    const observations = record?.motion_observations || {};
    const observedTimes = Array.isArray(observations.time_seconds) ? observations.time_seconds : [];
    const plottedTimes = Array.isArray(record?.time_seconds) ? record.time_seconds : [];
    const speed = Array.isArray(observations.source_sog) ? observations.source_sog : observations.sog_kn;
    const course = Array.isArray(observations.source_cog) ? observations.source_cog : observations.cog_deg;
    const speedMask = observations.sog_reported_mask || observations.reported_mask;
    const courseMask = observations.cog_reported_mask || observations.reported_mask;
    function observed(values, mask, index, upperBound) {
      if (index < 0 || (Array.isArray(mask) && mask[index] !== true) || !valid(values?.[index])) return null;
      const value = Number(values[index]);
      return value >= 0 && value < upperBound ? value : null;
    }
    const sog = [], cog = [];
    for (const time of plottedTimes) {
      const matches = [];
      if (valid(time)) observedTimes.forEach((candidate, i) => { if (valid(candidate) && Math.abs(Number(candidate) - Number(time)) <= .001) matches.push(i); });
      // Ambiguous/missing timestamps cannot establish an original report.
      const index = matches.length === 1 ? matches[0] : -1;
      sog.push(observed(speed, speedMask, index, 102.2));
      cog.push(observed(course, courseMask, index, 360));
    }
    return {sog_kn: sog, cog_deg: cog, missing_sog_count: sog.filter(value => value === null).length,
      missing_cog_count: cog.filter(value => value === null).length};
  }

  function typeDistribution(counts, {priorityWindows, qualityWindows, regionalWindows} = {}) {
    return kinds.map(([key, label]) => {
      const count = valid(counts?.[key]) ? Math.max(0, Math.floor(Number(counts[key]))) : 0;
      const denominatorValue = key === 'congestion' ? regionalWindows : key === 'observation_glitch' ? qualityWindows : priorityWindows;
      const denominator = valid(denominatorValue) ? Math.max(0, Math.floor(Number(denominatorValue))) : null;
      const unit = key === 'congestion' ? '区域时间窗' : '船舶窗口';
      const basis = key === 'congestion' ? '占全部区域时间窗' : key === 'observation_glitch' ? '占全部已分析船舶窗口，可与行为项重叠' : '占全部优先核查船舶窗口';
      return {key, label, count, denominator, percent: denominator > 0 ? 100 * count / denominator : null, unit, basis};
    });
  }
  return {timePositions, seriesSegments, originalSeries, typeDistribution};
}));
