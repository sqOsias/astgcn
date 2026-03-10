const ctx = document.getElementById('chart').getContext('2d');
const chart = new Chart(ctx, { 
  type: 'line', // 折线图类型
  data: { 
    labels: [], // X轴标签（时间/次数）
    datasets: [{ 
      label: 'Node 0 Prediction', // 折线名称：0号节点预测值
      data: [], // Y轴数据（预测车速）
      borderColor: '#3e95cd' // 折线颜色（浅蓝色）
    }] 
  }, 
  options: { 
    animation: false, // 关闭动画（实时更新更流畅）
    responsive: true // 响应式（适配不同屏幕）
  } 
});
async function start() {
  // 1. 获取后端API地址（从页面输入框）
  const url = document.getElementById('backendUrl').value;
  let idx = 0; // 计数变量（作为X轴标签）
  
  // 2. 定时任务：每隔指定秒数执行一次
  setInterval(async () => {
    // 3. 生成模拟输入数据
    const N = 307; // 节点数（和后端一致：307个交通节点）
    const values = new Array(N).fill(0); // 生成307个0的数组
    
    // 4. 转换为后端期望的格式：(307, 1, 12) → 但代码里写的是(307, 1, 12)？
    // 注意：原代码这里有格式错误，正确的后端期望格式是 (N, F, T) = (307, 3, 12)（F=3：速度、小时、日期），但这里模拟为[[0,0,...12个0]]
    const input_data = values.map(v => [[v, v, v, v, v, v, v, v, v, v, v, v]]);
    
    // 5. 调用后端/predict接口
    const res = await fetch(url + '/predict', { 
      method: 'POST', 
      headers: { 'Content-Type': 'application/json' }, 
      body: JSON.stringify({ input_data }) 
    });
    
    // 6. 解析响应数据
    const js = await res.json();
    const v = js.predictions[0][0]; // 取0号节点的第一个预测值
    
    // 7. 更新图表数据
    chart.data.labels.push(idx++); // X轴添加计数（0,1,2...）
    chart.data.datasets[0].data.push(v); // Y轴添加预测值
    
    // 8. 只保留最近120个数据点（避免图表过长）
    if (chart.data.labels.length > 120) { 
      chart.data.labels.shift(); // 删除最旧的X轴标签
      chart.data.datasets[0].data.shift(); // 删除最旧的Y轴数据
    }
    
    // 9. 刷新图表
    chart.update();
  }, parseFloat(document.getElementById('interval').value) * 1000); // 间隔（秒→毫秒）
}
document.getElementById('startBtn').addEventListener('click', start);
