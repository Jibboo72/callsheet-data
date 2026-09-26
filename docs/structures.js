(function(){
  // Grades the same nightly props as singles and round robins, $33 a night each.
  var BUDGET=33, SIZES=[6,8,12], size=8, LOG=[];
  var STRUCTS=[{name:"Singles",k:[1]},{name:"2-pick round robin",k:[2]},
               {name:"3-pick round robin",k:[3]},{name:"2s + 3s round robin",k:[2,3]}];
  var $=function(id){return document.getElementById(id)};
  var money=function(v){return (v<0?"-$":"+$")+Math.abs(v).toFixed(0)};
  var am=function(p){return p>0?"+"+p:""+p};
  var esc=function(s){return String(s).replace(/[&<>"]/g,function(c){return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]})};
  var label={goal:"Anytime goal",assist:"1+ assist"};
  function dec(p){return p>0?1+p/100:1+100/Math.abs(p)}

  function combos(n,k){
    var out=[];
    (function rec(start,acc){
      if(acc.length===k){out.push(acc.slice());return}
      for(var i=start;i<n;i++){acc.push(i);rec(i+1,acc);acc.pop()}
    })(0,[]);
    return out;
  }
  function count(n,ks){return ks.reduce(function(t,k){return t+(n>=k?combos(n,k).length:0)},0)}

  // Top props by edge: one per player, at most two per game, flagged plays left out.
  function slate(bets){
    var players={}, games={}, out=[];
    bets.filter(function(b){return b.kind==="bet"&&b.price!=null&&!b.flag})
      .sort(function(a,b){return b.ev-a.ev}).forEach(function(b){
        if(out.length>=size||players[b.player_id]||(games[b.game_id]||0)>=2) return;
        players[b.player_id]=1; games[b.game_id]=(games[b.game_id]||0)+1; out.push(b);
      });
    return out;
  }
  function nightProfit(legs,ks){
    var cs=[]; ks.forEach(function(k){if(legs.length>=k) cs=cs.concat(combos(legs.length,k))});
    if(!cs.length) return null;
    var stake=BUDGET/cs.length, ret=0;
    cs.forEach(function(c){var r=1; c.forEach(function(i){r*=legs[i]}); ret+=r*stake});
    return ret-BUDGET;
  }

  function render(){
    var chips=$("sizes"); chips.innerHTML="";
    SIZES.forEach(function(n){
      var b=document.createElement("button"); b.textContent=n+" props";
      b.setAttribute("aria-pressed", n===size?"true":"false");
      b.onclick=function(){size=n; render()};
      chips.appendChild(b);
    });
    var byDate={};
    LOG.forEach(function(b){(byDate[b.date]=byDate[b.date]||[]).push(b)});
    var dates=Object.keys(byDate).sort(), graded=[], pending=null;
    dates.forEach(function(d){
      var s=slate(byDate[d]);
      if(!s.length) return;
      if(s.some(function(b){return b.status==="pending"})) pending={date:d, legs:s};
      else graded.push(s.map(function(b){return b.status==="win"?dec(b.price):b.status==="void"?1:0}));
    });

    var tb=$("structs"); tb.innerHTML="";
    var rows=STRUCTS.map(function(st){
      var cum=0, peak=0, slump=0, wins=0, nights=0;
      graded.forEach(function(legs){
        var p=nightProfit(legs,st.k); if(p===null) return;
        nights++; cum+=p; if(p>0) wins++;
        peak=Math.max(peak,cum); slump=Math.max(slump,peak-cum);
      });
      return {st:st, profit:cum, nights:nights, wins:wins, slump:slump};
    });
    var best=rows.reduce(function(a,r){return r.nights&&(!a||r.profit>a.profit)?r:a},null);
    rows.forEach(function(r){
      var tr=document.createElement("tr");
      var roi=r.nights?r.profit/(r.nights*BUDGET):null;
      tr.innerHTML="<td"+(r===best&&graded.length?" class='lead'":"")+">"+r.st.name+"</td>"+
        "<td class='"+(r.profit>=0?"up":"down")+"'>"+(r.nights?money(r.profit):"—")+"</td>"+
        "<td>"+(roi==null?"—":(roi>0?"+":"")+(roi*100).toFixed(1)+"%")+"</td>"+
        "<td>"+(r.nights?r.wins+"/"+r.nights:"—")+"</td>"+
        "<td>"+(r.nights?"$"+r.slump.toFixed(0):"—")+"</td>";
      tb.appendChild(tr);
    });
    $("structNote").textContent = !graded.length
      ? "Nothing graded yet. The first results show up the day after the first priced night."
      : graded.length<20
      ? graded.length+" night"+(graded.length===1?"":"s")+" graded. The gaps between structures mean little before about 20 nights."
      : graded.length+" nights graded.";

    var ol=$("slate"); ol.innerHTML="";
    if(!pending){$("slateMeta").textContent="No slate waiting on results."; return}
    var n=pending.legs.length;
    $("slateMeta").textContent=n+" props for "+pending.date+". 2-pick round robin: "+count(n,[2])+" bets at $"+
      (count(n,[2])?(BUDGET/count(n,[2])).toFixed(2):"0")+". 3-pick: "+count(n,[3])+" bets at $"+
      (count(n,[3])?(BUDGET/count(n,[3])).toFixed(2):"0")+".";
    pending.legs.forEach(function(b){
      var li=document.createElement("li"); li.className="play";
      li.innerHTML="<span class='who'>"+esc(b.name)+"</span><span class='game'>"+b.team+" vs "+b.opp+", "+label[b.market]+"</span>"+
        "<span class='nums'>Logged at <b>"+am(b.price)+"</b> "+esc(b.book||"")+"</span>"+
        "<span class='ev'><b>"+(b.ev>0?"+":"")+(b.ev*100).toFixed(0)+"%</b><span>EV</span></span>";
      ol.appendChild(li);
    });
  }

  fetch("data/sim_log.json",{cache:"no-store"})
    .then(function(r){if(!r.ok) throw 0; return r.json()})
    .then(function(log){
      LOG=log;
      if(!log.some(function(b){return b.kind==="bet"})) return;
      $("structBlock").hidden=false; render();
    })
    .catch(function(){});
})();
