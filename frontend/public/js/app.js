const ctx = document.getElementById('chart').getContext('2d');
const chart = new Chart(ctx,{type:'line',data:{labels:[],datasets:[{label:'Node 0 Prediction',data:[],borderColor:'#3e95cd'}]},options:{animation:false,responsive:true}});
async function start(){
  const url=document.getElementById('backendUrl').value;
  let idx=0;
  setInterval(async ()=>{
    const N=307;
    const values=new Array(N).fill(0);
    // 转换为后端期望的格式：(N_nodes, F_features, T_timestamps) = (307, 1, 12)
    const input_data = values.map(v => [[v, v, v, v, v, v, v, v, v, v, v, v]]);
    const res=await fetch(url+'/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({input_data})});
    const js=await res.json();
    const v=js.predictions[0][0];
    chart.data.labels.push(idx++);
    chart.data.datasets[0].data.push(v);
    if(chart.data.labels.length>120){chart.data.labels.shift();chart.data.datasets[0].data.shift();}
    chart.update();
  },parseFloat(document.getElementById('interval').value)*1000);
}
document.getElementById('startBtn').addEventListener('click',start);
