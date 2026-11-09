'use strict';
const node=(tag,text,cls)=>{const e=document.createElement(tag);e.textContent=text;if(cls)e.className=cls;return e;};
fetch('/api/projects').then(r=>{if(!r.ok)throw Error('Cannot load projects');return r.json();}).then(data=>{
 const list=document.getElementById('projects');list.replaceChildren();
 for(const p of data.projects){const card=node('section','','project-card');card.append(node('div',p.demo?'DEMO / SYNTHETIC':'TRAJECTORY REVIEW','eyebrow'),node('h2',p.title),node('p',p.description));
 const meta=node('div',`${p.cases.length} clips · ${p.reviewed} reviews saved`,'card-meta');const a=node('a','Open trajectory review →','button primary');a.href='/trajectory-review/?project='+encodeURIComponent(p.id);card.append(meta,a);list.append(card);}
 if(data.legacy_url){const card=node('section','','project-card legacy');card.append(node('div','LEADER RELATION REVIEW','eyebrow'),node('h2','Leader relation review'),node('p','Review target–leader pairs, absent or hidden leaders, and changes within a clip. Existing records remain available.'));const a=node('a','Open leader review →','button');a.href=data.legacy_url;card.append(node('div','Existing project and annotations retained','card-meta'),a);list.append(card);}
}).catch(e=>{document.getElementById('projects').textContent=e.message;});
