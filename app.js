"use strict";
const clips = {
  fall: {src:"assets/fall-detection.mp4",poster:"assets/fall-poster.jpg",title:"跌倒动作识别",meta:"3.3 秒 · 100 帧",description:"动作开始阶段有短暂判断抖动；素材时间 0.934 秒后持续判为跌倒动作。该时间不是报警延迟。"},
  walk: {src:"assets/walking.mp4",poster:"assets/walking-poster.jpg",title:"正常走路识别",meta:"10 秒 · 300 帧",description:"这段走路录像在 285 个可判断时点中没有跌倒误报。开头 15 帧用于收集画面；单段结果不能代表通用误报率。"}
};
const player=document.getElementById("player");
const errorMessage=document.getElementById("player-error");
document.querySelectorAll("[data-clip]").forEach(button=>{
  button.addEventListener("click",()=>{
    const clip=clips[button.dataset.clip];
    if(button.getAttribute("aria-pressed")==="true")return;
    player.pause();
    player.poster=clip.poster;
    player.querySelector("source").src=clip.src;
    player.setAttribute("aria-label",`${clip.title}视频，无音频，画面包含中文识别状态与分数`);
    errorMessage.hidden=true;
    player.load();
    document.getElementById("clip-title").textContent=clip.title;
    document.getElementById("clip-meta").textContent=clip.meta;
    document.getElementById("clip-description").textContent=clip.description;
    document.querySelectorAll("[data-clip]").forEach(item=>{
      const selected=item===button;
      item.setAttribute("aria-pressed",String(selected));
      item.classList.toggle("selected",selected);
    });
  });
});
player.addEventListener("error",()=>{errorMessage.hidden=false;});
player.querySelector("source").addEventListener("error",()=>{errorMessage.hidden=false;});
document.getElementById("share-button").addEventListener("click",async()=>{
  const status=document.getElementById("share-status");
  const url=new URL(window.location.href);url.hash="";
  try{await navigator.clipboard.writeText(url.href);status.textContent="网页链接已复制，可以粘贴转发。";}
  catch{status.textContent=`请复制浏览器地址栏链接：${url.href}`;}
});
