export function renderMath(root=document){
  const nodes=[];
  if(root instanceof Element && root.matches("[data-katex]"))nodes.push(root);
  nodes.push(...root.querySelectorAll("[data-katex]"));
  if(!window.katex)return false;
  nodes.forEach(node=>{
    if(node.dataset.katexRendered==="true")return;
    const source=node.textContent.trim();
    window.katex.render(source,node,{
      displayMode:node.dataset.katex!=="inline",
      throwOnError:false,
      strict:"ignore",
      trust:false
    });
    node.dataset.katexRendered="true";
  });
  return true;
}

export function renderMathWhenReady(root=document){
  if(renderMath(root))return;
  window.addEventListener("load",()=>renderMath(root),{once:true});
}
