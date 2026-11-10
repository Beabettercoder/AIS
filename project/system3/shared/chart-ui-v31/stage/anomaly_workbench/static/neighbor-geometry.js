/* Pure geometry selection: no invented vessels and no interpolated reports. */
(function(root){
  'use strict';
  const RADIUS_NM=3.5;
  const position=p=>Array.isArray(p)&&p.length===2&&p.every(Number.isFinite)&&Math.abs(p[0])<=90&&Math.abs(p[1])<=180;
  function geographicDistanceNm(a,b){
    if(!position(a)||!position(b))return null;
    const rad=Math.PI/180,deltaLat=(b[0]-a[0])*rad,deltaLon=((b[1]-a[1]+540)%360-180)*rad;
    const h=Math.sin(deltaLat/2)**2+Math.cos(a[0]*rad)*Math.cos(b[0]*rad)*Math.sin(deltaLon/2)**2;
    return 6371008.8*2*Math.atan2(Math.sqrt(Math.max(0,h)),Math.sqrt(Math.max(0,1-h)))/1852;
  }
  function distanceNm(record,step,neighbor){
    const reported=neighbor.distance_nm;
    if(reported!==null&&reported!==undefined&&Number.isFinite(Number(reported))&&Number(reported)>=0)return Number(reported);
    return geographicDistanceNm(record.latlon?.[step],neighbor.position);
  }
  function frameAt(record,step){
    const last=Math.max(0,(record.latlon||[]).length-1),k=Math.max(0,Math.min(last,Math.trunc(step)));
    if(Array.isArray(record.neighbor_frames)&&record.neighbor_frames[k])return record.neighbor_frames[k];
    // A legacy final snapshot is never repeated at earlier times.
    return {step:k,time_epoch:Date.parse(record.start_utc)/1000+(record.time_seconds?.[k]||0),
      time_utc:null,direction_basis:'末步快照',neighbors:k===last?(record.neighbor_tracks||[]):[],count:k===last?(record.neighbor_tracks||[]).length:0};
  }
  function busiestStep(record){
    const last=Math.max(0,(record.latlon||[]).length-1);
    let best=last,count=0;
    for(let i=0;i<=last;i++){const n=visibleNeighbors(record,i).length;if(n>0&&n>=count){best=i;count=n;}}
    return best;
  }
  function preferredStep(record,requested){
    const last=Math.max(0,(record.latlon||[]).length-1);
    if(requested!==null&&requested!==undefined&&Number.isFinite(Number(requested)))return Math.max(0,Math.min(last,Math.trunc(Number(requested))));
    return visibleNeighbors(record,last).length?last:busiestStep(record);
  }
  function visibleNeighbors(record,step){
    const frame=frameAt(record,step),tracks=new Map((record.neighbor_track_catalog||[]).map(t=>[String(t.mmsi),t]));
    return (frame.neighbors||[]).filter(n=>{
      const distance=distanceNm(record,step,n),actual=geographicDistanceNm(record.latlon?.[step],n.position);
      return position(n.position)&&String(n.mmsi)!==String(record.vessel_id)&&distance!==null&&distance<=RADIUS_NM&&(actual===null||actual<=RADIUS_NM);
    }).map(n=>{
      const history=tracks.get(String(n.mmsi)),target=record.latlon?.[step];let source=[];
      if(history)source=(history.latlon||[]).map((p,i)=>({position:p,time:history.timestamps?.[i]})).filter(p=>Number.isFinite(p.time)&&p.time<=frame.time_epoch);
      else if(step===(record.latlon||[]).length-1)source=(n.latlon||[]).map(p=>({position:p}));
      const segments=[];let run=[];
      for(const p of source){
        const distance=geographicDistanceNm(target,p.position);
        if(position(p.position)&&distance!==null&&distance<=RADIUS_NM)run.push(p.position);
        else if(run.length){segments.push(run);run=[];}
      }
      if(!run.length||run.at(-1)[0]!==n.position[0]||run.at(-1)[1]!==n.position[1])run.push(n.position);
      if(run.length)segments.push(run);
      return {...n,distance_nm:distanceNm(record,step,n),visibleSegments:segments,visibleTrack:segments.flat()};
    });
  }
  function boundsPoints(record,step,includeNeighbors=true){
    const points=(record.repaired_latlon||record.latlon||[]).filter(position).slice();
    if(includeNeighbors)for(const n of visibleNeighbors(record,step))points.push(...n.visibleTrack,n.position);
    return points;
  }
  const api={frameAt,preferredStep,busiestStep,visibleNeighbors,boundsPoints,distanceNm,geographicDistanceNm,radiusNm:RADIUS_NM};
  root.AISNeighborGeometry=api;
  if(typeof module==='object'&&module.exports)module.exports=api;
})(typeof window!=='undefined'?window:globalThis);
