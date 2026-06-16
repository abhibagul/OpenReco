// icons.js — functional line-art icon set + OpenReco mesh mark.
// Icons are 24x24 stroke paths using currentColor. Injected into [data-icon] nodes.
(function(){
  var S = function(inner){ return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg">'+inner+'</svg>'; };
  var ICONS = {
    photos: S('<rect x="3" y="3" width="14" height="14" rx="2"/><path d="M3 13l3.5-3.5a2 2 0 0 1 2.8 0L14 14"/><circle cx="13" cy="8" r="1.4"/><path d="M21 7v12a2 2 0 0 1-2 2H7"/>'),
    align: S('<circle cx="5" cy="6" r="1.6"/><circle cx="19" cy="5" r="1.6"/><circle cx="12" cy="19" r="1.6"/><circle cx="17" cy="13" r="1.6"/><path d="M6.4 6.7l9.4 5.7M17.6 5.8L13 17.4M6 7l5.6 10.6"/>'),
    dense: S('<circle cx="6" cy="6" r="1.1"/><circle cx="12" cy="5" r="1.1"/><circle cx="18" cy="7" r="1.1"/><circle cx="5" cy="12" r="1.1"/><circle cx="11" cy="11" r="1.1"/><circle cx="17" cy="13" r="1.1"/><circle cx="8" cy="17" r="1.1"/><circle cx="14" cy="18" r="1.1"/><circle cx="19" cy="18" r="1.1"/>'),
    model: S('<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3z"/><path d="M12 3v18M4 7.5l8 4.5 8-4.5"/>'),
    texture: S('<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M3 15h18M9 3v18M15 3v18"/>'),
    dem: S('<path d="M3 17l5-7 4 4 4-6 5 7"/><path d="M3 21h18"/>'),
    ortho: S('<rect x="3" y="3" width="18" height="18" rx="1.5"/><path d="M3 9h18M3 15h18M9 3v18M15 3v18"/><path d="M3 3l6 6M15 9l6-6" opacity=".5"/>'),
    veg: S('<path d="M12 21c0-6 0-9 4-12 0 5-1 8-4 9.5"/><path d="M12 21c0-5-1-7-4-9.5 0 4 1 6.5 4 8"/><path d="M12 21V11"/>'),
    classify: S('<path d="M3 7l9-4 9 4-9 4-9-4z"/><path d="M3 12l9 4 9-4M3 17l9 4 9-4"/>'),
    tiled: S('<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>'),
    contours: S('<path d="M4 13a8 5 0 0 1 16 0"/><path d="M6.5 13a5.5 3.4 0 0 1 11 0"/><path d="M9 13a3 1.8 0 0 1 6 0"/>'),
    cleancloud: S('<circle cx="8" cy="9" r="1"/><circle cx="13" cy="7" r="1"/><circle cx="11" cy="12" r="1"/><circle cx="7" cy="14" r="1"/><path d="M16 5l1 2.4L19.5 8.4 17 9.5 16 12l-1-2.5L12.5 8.4 15 7.4 16 5z"/>'),
    cleanmesh: S('<circle cx="7" cy="7" r="2.4"/><path d="M9 9l10 10M19 9L9 19" opacity=".55"/><path d="M14.5 12.5l5.5-1.5M12.5 14.5l-1.5 5.5"/>'),
    georef: S('<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c2.6 2.6 2.6 15.4 0 18M12 3c-2.6 2.6-2.6 15.4 0 18"/>'),
    merge: S('<path d="M7 4v5a4 4 0 0 0 4 4h6"/><path d="M14 9l3-3-3-3M7 20v-3"/><circle cx="7" cy="19.5" r="1.4"/>'),
    // feature / viz icons
    measure: S('<path d="M14.5 3.5l6 6L9 21l-6-6L14.5 3.5z"/><path d="M7 11l2 2M10 8l2 2M13 5l2 2"/>'),
    layers: S('<path d="M12 3l9 5-9 5-9-5 9-5z"/><path d="M3 13l9 5 9-5"/>'),
    color: S('<circle cx="12" cy="12" r="9"/><circle cx="9" cy="9" r="1.2"/><circle cx="15" cy="9" r="1.2"/><circle cx="16" cy="14" r="1.2"/><path d="M12 21a3 3 0 0 1 0-6 2 2 0 0 0 0-4"/>'),
    eye: S('<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="2.6"/>'),
    gpu: S('<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="8" y="8" width="8" height="8" rx="1"/><path d="M9 4V2M15 4V2M9 22v-2M15 22v-2M4 9H2M4 15H2M22 9h-2M22 15h-2"/>'),
    cache: S('<path d="M21 12a9 9 0 1 1-3-6.7"/><path d="M21 4v5h-5"/>'),
    crs: S('<circle cx="12" cy="10" r="3"/><path d="M12 21c5-5.5 7-8.4 7-11a7 7 0 1 0-14 0c0 2.6 2 5.5 7 11z"/>'),
    sliders: S('<path d="M4 6h10M18 6h2M4 12h2M10 12h10M4 18h7M15 18h5"/><circle cx="16" cy="6" r="2"/><circle cx="8" cy="12" r="2"/><circle cx="13" cy="18" r="2"/>'),
    check: S('<path d="M20 6L9 17l-5-5"/>'),
    bolt: S('<path d="M13 2L4 14h7l-1 8 9-12h-7l1-8z"/>'),
    open: S('<path d="M9 3H4v5M15 3h5v5M20 16v5h-5M9 21H4v-5"/>'),
    download: S('<path d="M12 3v12M7 10l5 5 5-5M5 21h14"/>'),
    github: S('<path d="M9 19c-4 1.5-4-2.5-6-3m12 5v-3.5c0-1 .1-1.4-.5-2 2.8-.3 5.5-1.4 5.5-6a4.6 4.6 0 0 0-1.3-3.2 4.3 4.3 0 0 0-.1-3.2s-1.1-.3-3.5 1.3a12 12 0 0 0-6.2 0C6.5 2.8 5.4 3.1 5.4 3.1a4.3 4.3 0 0 0-.1 3.2A4.6 4.6 0 0 0 4 9.5c0 4.6 2.7 5.7 5.5 6-.6.6-.6 1.2-.5 2V21"/>'),
    arrow: S('<path d="M5 12h14M13 6l6 6-6 6"/>'),
    apple: S('<path d="M16 3c.2 1.4-.4 2.7-1.2 3.6-.8.9-2.1 1.6-3.3 1.5-.2-1.3.5-2.7 1.2-3.5C13.4 3.6 14.9 3 16 3z"/><path d="M19 16.5c-.6 1.4-.9 2-1.7 3.2-1.1 1.7-2.6 3.8-4.5 3.3-1.7-.5-2.1-1.1-3.5-1.1-1.4 0-1.9.6-3.5 1.1-1.9.5-3.4-1.6-4.5-3.3C-1 15.6-.2 9.5 3.4 9.5c1.5 0 2.4 1 3.6 1s1.8-1 3.6-1c1.4 0 2.7.8 3.6 2-3.2 1.8-2.7 6.3 1.2 5z"/>'),
    win: S('<path d="M3 5l8-1v8H3V5zM13 3.7L21 3v9h-8V3.7zM3 13h8v6l-8-1v-5zM13 13h8v8l-8-1v-7z"/>'),
    linux: S('<path d="M9 4c-1 1-1.2 3-1 5 .1 1.5-1 2.5-1.6 4-.6 1.5.3 3 1.6 4 1 .8 1 2 4 2s3-1.2 4-2c1.3-1 2.2-2.5 1.6-4-.6-1.5-1.7-2.5-1.6-4 .2-2 0-4-1-5a3 3 0 0 0-6 0z"/><circle cx="10.5" cy="9" r=".6" fill="currentColor"/><circle cx="13.5" cy="9" r=".6" fill="currentColor"/>'),
    target: S('<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.4" fill="currentColor"/>'),
    chunks: S('<path d="M12 2l9 5v10l-9 5-9-5V7l9-5z"/><path d="M12 2v10M3 7l9 5 9-5" opacity=".5"/>'),
    map: S('<path d="M9 3L3 5v16l6-2 6 2 6-2V3l-6 2-6-2z"/><path d="M9 3v16M15 5v16"/>')
  };
  function meshSVG(variant){
    var T=[50,12],UR=[84.6,32],LR=[84.6,72],B=[50,92],LL=[15.4,72],UL=[15.4,32],C=[50,52];
    var hull=[T,UR,LR,B,LL,UL];
    var P={
      'default':{f1:'#7FB0FF',f2:'#3B82F6',f3:'#1D4ED8',edge:'#1D4ED8',wire:'#9DB8E6',ctr:'#3B82F6'},
      'ondark' :{f1:'#7FB0FF',f2:'#3B82F6',f3:'#1D4ED8',edge:'#7FB0FF',wire:'#3B5B8F',ctr:'#3B82F6'}
    }[variant||'default'];
    function pts(a,b){return a[0]+','+a[1]+' '+b[0]+','+b[1]+' '+C[0]+','+C[1];}
    var s='<svg viewBox="0 0 100 100" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">';
    s+='<polygon points="'+pts(UL,T)+'" fill="'+P.f1+'"/>';
    s+='<polygon points="'+pts(T,UR)+'" fill="'+P.f2+'"/>';
    s+='<polygon points="'+pts(UR,LR)+'" fill="'+P.f3+'"/>';
    [LR,B,LL,UL].forEach(function(q){s+='<line x1="'+C[0]+'" y1="'+C[1]+'" x2="'+q[0]+'" y2="'+q[1]+'" stroke="'+P.wire+'" stroke-width="2" stroke-linecap="round"/>';});
    s+='<polygon points="'+hull.map(function(q){return q[0]+','+q[1];}).join(' ')+'" fill="none" stroke="'+P.edge+'" stroke-width="3" stroke-linejoin="round"/>';
    hull.forEach(function(q){s+='<circle cx="'+q[0]+'" cy="'+q[1]+'" r="3.4" fill="'+P.edge+'"/>';});
    s+='<circle cx="'+C[0]+'" cy="'+C[1]+'" r="3.4" fill="'+P.ctr+'"/>';
    return s+'</svg>';
  }
  document.querySelectorAll('[data-icon]').forEach(function(el){
    var k=el.getAttribute('data-icon'); if(ICONS[k]) el.innerHTML=ICONS[k];
  });
  document.querySelectorAll('[data-mark]').forEach(function(el){ el.innerHTML=meshSVG(el.getAttribute('data-mark')); });

  // scroll reveal (+ stagger for grids)
  var io=new IntersectionObserver(function(es){es.forEach(function(e){
    if(e.isIntersecting){
      var el=e.target;
      if(el.classList.contains('stagger')){
        [].slice.call(el.children).forEach(function(c,i){ c.style.transitionDelay=(i*55)+'ms'; });
      }
      el.classList.add('in');
      io.unobserve(el);
    }
  });},{threshold:.12});
  document.querySelectorAll('.reveal,.stagger').forEach(function(el){io.observe(el);});
})();
