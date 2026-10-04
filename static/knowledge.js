const el=(tag,text)=>{const x=document.createElement(tag);if(text!==undefined)x.textContent=text;return x;};
document.getElementById('research-example').onclick=()=>{document.getElementById('question').value='图像质量分数、检测框数量及分数上升能代替真实框标注下的召回与误报评测吗？';document.getElementById('question').focus();};
document.getElementById('search-form').onsubmit=async event=>{
 event.preventDefault();const button=document.getElementById('search'),status=document.getElementById('status');button.disabled=true;status.textContent='正在本地检索，首次加载模型可能稍慢…';
 try{const response=await fetch('/api/knowledge/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:document.getElementById('question').value,vision_mode:document.getElementById('vision').value,top_k:Number(document.getElementById('top-k').value)})});const data=await response.json();if(!response.ok)throw new Error(data.error);
 const panel=document.getElementById('rankings');panel.replaceChildren();
 for(const [mode,result] of Object.entries(data.comparisons)){const card=el('section');card.className='card';card.append(el('h2',({tfidf:'TF-IDF 词语匹配',embedding:'Embedding 语义检索',hybrid:'Hybrid 排名融合'}[mode]||mode)));
 if(!result.available){card.append(el('p','不可用：'+result.error));panel.append(card);continue;}
 card.append(el('p',`查询 ${result.query_ms.toFixed(1)} ms · 加载 ${result.load_ms.toFixed(1)} ms`));card.append(el('p',`语料 ${result.metadata.corpus} · 版本 ${result.metadata.corpus_hash.slice(0,12)}`));
 if(!result.results.length)card.append(el('p',result.empty_reason));
 result.results.forEach((r,i)=>{const item=el('article');item.append(el('h3',`${i+1}. ${r.title}`));item.append(el('p',`类型：${r.source_type||'历史资料'} · 核实：${r.verification_level||'项目文件'}`));if(r.version_warning)item.append(el('p',r.version_warning));item.append(el('p',`${r.score_kind||'tfidf_cosine'}：${r.score}（非概率）`));if(r.tfidf_score!=null)item.append(el('p',`TF-IDF ${r.tfidf_score} · Embedding ${r.embedding_score==null?'未计算':r.embedding_score}`));item.append(el('p',r.text));const link=el('a','查看出处');link.href=r.source;link.target='_blank';link.rel='noopener noreferrer';item.append(link,el('p',`引用 ${r.citation_id} · 源位置 ${r.source_start==null?'旧分段':r.source_start+'—'+r.source_end}`));card.append(item);});panel.append(card);}
 status.textContent='对照完成；没有调用付费 API。';
 }catch(error){status.textContent=error.message;}finally{button.disabled=false;}
};
