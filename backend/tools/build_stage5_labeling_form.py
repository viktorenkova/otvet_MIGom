from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACK = ROOT / ".work/stage5-blind-review-pack.json"
DEFAULT_OUTPUT = ROOT / ".work/stage5-independent-labeling-form.html"


HTML = r'''<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:;">
  <title>Независимая разметка MIGTORG — этап 5</title>
  <style>
    :root { --ink:#172033; --muted:#637083; --line:#d8deea; --paper:#fff; --bg:#f4f7fb; --brand:#2457d6; --brand2:#e8efff; --good:#147a50; --warn:#a35a00; --bad:#b42318; }
    * { box-sizing:border-box; }
    body { margin:0; font:15px/1.45 Inter,Segoe UI,Arial,sans-serif; color:var(--ink); background:var(--bg); }
    header { position:sticky; top:0; z-index:5; background:#122650; color:#fff; padding:14px 22px; box-shadow:0 2px 12px #0c193633; }
    header h1 { margin:0 0 4px; font-size:20px; }
    header p { margin:0; color:#d9e4ff; font-size:13px; }
    .layout { display:grid; grid-template-columns:310px minmax(0,1fr); gap:18px; max-width:1500px; margin:18px auto; padding:0 18px 30px; }
    .panel { background:var(--paper); border:1px solid var(--line); border-radius:12px; box-shadow:0 3px 14px #23395d12; }
    aside { padding:16px; align-self:start; position:sticky; top:88px; max-height:calc(100vh - 106px); overflow:auto; }
    main { padding:20px 24px 28px; min-width:0; }
    h2 { margin:0 0 14px; font-size:20px; }
    h3 { margin:22px 0 10px; font-size:16px; }
    .notice { border-left:4px solid var(--brand); background:var(--brand2); padding:11px 13px; margin:0 0 16px; border-radius:6px; }
    .warning { border-left-color:var(--warn); background:#fff4df; }
    .grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:13px 16px; }
    .field { display:flex; flex-direction:column; gap:5px; min-width:0; }
    .field.full { grid-column:1/-1; }
    label, .label { font-weight:650; font-size:13px; }
    .hint { color:var(--muted); font-size:12px; font-weight:400; }
    input[type=text], select, textarea { width:100%; border:1px solid #b9c3d5; border-radius:7px; background:#fff; color:var(--ink); padding:9px 10px; font:inherit; }
    textarea { min-height:82px; resize:vertical; }
    input:focus, select:focus, textarea:focus { outline:3px solid #2457d622; border-color:var(--brand); }
    .check { display:flex; align-items:flex-start; gap:8px; font-weight:500; }
    .check input { margin-top:4px; }
    .message { background:#f8faff; border:1px solid #ccd7ed; border-radius:10px; padding:15px; font-size:17px; white-space:pre-wrap; overflow-wrap:anywhere; }
    .context { display:grid; gap:8px; margin-bottom:10px; }
    .turn { border-left:3px solid #9fb2d8; padding:8px 10px; background:#f8f9fc; border-radius:4px; }
    .turn.current { border-left-color:var(--brand); background:var(--brand2); }
    .meta { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px; color:var(--muted); font-size:13px; }
    .tag { background:#edf1f7; border-radius:999px; padding:3px 9px; color:#344054; }
    .progress { height:9px; background:#e6eaf1; border-radius:999px; overflow:hidden; margin:7px 0; }
    .progress span { display:block; height:100%; background:var(--good); width:0; }
    .stats { font-size:13px; color:var(--muted); margin-bottom:14px; }
    .actions { display:flex; flex-wrap:wrap; gap:9px; margin-top:22px; padding-top:16px; border-top:1px solid var(--line); }
    button { border:0; border-radius:7px; padding:9px 13px; font:650 14px/1.2 inherit; cursor:pointer; background:#e9edf4; color:#27364f; }
    button.primary { background:var(--brand); color:#fff; }
    button.good { background:var(--good); color:#fff; }
    button.warn { background:#fff0d5; color:#7a4300; }
    button:disabled { opacity:.45; cursor:not-allowed; }
    .nav { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin:12px 0 16px; }
    .status { padding:9px 10px; border-radius:7px; font-size:13px; margin:10px 0; background:#eef2f7; }
    .status.good { background:#e7f6ef; color:#0c6842; }
    .status.bad { background:#ffebe9; color:#9d2118; }
    .status.warn { background:#fff3dc; color:#8a4a00; }
    .hidden { display:none !important; }
    .small { font-size:12px; color:var(--muted); }
    .divider { border-top:1px solid var(--line); margin:16px 0; }
    @media (max-width:900px) { .layout { grid-template-columns:1fr; } aside { position:static; max-height:none; } .grid { grid-template-columns:1fr; } }
  </style>
</head>
<body>
<header><h1>Независимая разметка MIGTORG</h1><p>Закрытый blind-набор этапа 5 · ответы бота в форме отсутствуют</p></header>
<div class="layout">
  <aside class="panel">
    <h2>Работа с набором</h2>
    <div id="datasetMeta" class="small"></div>
    <div class="progress"><span id="progressBar"></span></div>
    <div id="progressText" class="stats"></div>

    <div class="field"><label for="reviewerId">Идентификатор исполнителя</label><input id="reviewerId" type="text" placeholder="Например: support-qa-01"></div>
    <div style="height:10px"></div>
    <label class="check"><input id="independent" type="checkbox">Я не участвовал(а) в настройке router, retrieval, reranker, matching configs или answer contracts.</label>
    <label class="check"><input id="confidential" type="checkbox">Я не передаю контрольные обращения разработчику до первого blind-прогона.</label>
    <label class="check"><input id="privacy" type="checkbox">Я проверил(а), что в размеченных записях нет персональных данных.</label>

    <div class="divider"></div>
    <div class="field"><label for="filter">Показывать</label><select id="filter"><option value="all">Все записи</option><option value="pending">Только незавершённые</option><option value="needs_review">Требуют эксперта</option><option value="single">Одиночные обращения</option><option value="dialogue">Диалоги</option></select></div>
    <div class="field"><label for="jump">Перейти к ID</label><input id="jump" type="text" placeholder="blind-single-0001"></div>
    <div class="nav"><button id="prev">← Назад</button><button id="next">Вперёд →</button></div>
    <button id="saveDraft" style="width:100%">Скачать черновик JSON</button>
    <div style="height:8px"></div>
    <button id="exportFinal" class="good" style="width:100%">Проверить и скачать финальный JSON</button>
    <div style="height:8px"></div>
    <label class="small">Загрузить ранее сохранённый черновик<input id="importFile" type="file" accept="application/json,.json" style="display:block;margin-top:5px;max-width:100%"></label>
    <div id="globalStatus" class="status">Изменения автоматически сохраняются только в этом браузере.</div>
  </aside>

  <main class="panel">
    <div class="notice warning"><strong>Важно:</strong> определяйте правильное поведение по правилам MIGTORG. Не запускайте бота и не подстраивайте метку под его фактический ответ.</div>
    <div id="recordMeta" class="meta"></div>
    <div id="dialogueContext" class="context hidden"></div>
    <div id="message" class="message"></div>

    <h3>Правильное поведение бота</h3>
    <div class="grid">
      <div class="field"><label for="primaryTopic">Основная тема *</label><input id="primaryTopic" type="text" placeholder="Например: платежи"></div>
      <div class="field"><label for="specificSituation">Конкретная ситуация *</label><input id="specificSituation" type="text" placeholder="Например: списание без зачисления"></div>
      <div class="field"><label for="botAction">Действие бота *</label><select id="botAction"><option value="">— выберите —</option><option value="answer">Дать ответ</option><option value="clarify">Задать уточняющий вопрос</option><option value="support">Предложить обращение в поддержку</option><option value="out_of_scope">Сообщить, что вопрос не относится к MIGTORG</option><option value="safe_refusal">Безопасно отказать</option></select></div>
      <div class="field"><label for="confidence">Уверенность *</label><select id="confidence"><option value="">— выберите —</option><option value="high">Высокая</option><option value="medium">Средняя</option><option value="requires_review">Требуется экспертная проверка</option></select></div>

      <div class="field full"><label for="requiredInfo">Что обязательно сообщить <span class="hint">по одному пункту на строку</span></label><textarea id="requiredInfo"></textarea><label class="check"><input id="noRequired" type="checkbox">Обязательных сведений нет</label></div>
      <div class="field full"><label for="forbiddenInfo">Что запрещено сообщать <span class="hint">по одному пункту на строку</span></label><textarea id="forbiddenInfo"></textarea><label class="check"><input id="noForbidden" type="checkbox">Специальных запретов нет</label></div>
      <div class="field full"><label class="check"><input id="multipleValid" type="checkbox">Допустимо несколько правильных вариантов ответа</label><textarea id="alternatives" placeholder="Перечислите допустимые варианты — по одному на строку"></textarea></div>
    </div>

    <div id="dialogueFields" class="hidden">
      <h3>Контекст этого хода диалога</h3>
      <div class="grid">
        <div class="field"><label for="continuesTopic">Продолжает предыдущую тему? *</label><select id="continuesTopic"><option value="">— выберите —</option><option value="true">Да</option><option value="false">Нет, новая тема</option></select></div>
        <div class="field"><label for="resolvedAfter">Вопрос решён после этого хода? *</label><select id="resolvedAfter"><option value="">— выберите —</option><option value="true">Да</option><option value="false">Нет</option></select></div>
        <div class="field"><label for="supportHandoff">Нужна передача сотруднику? *</label><select id="supportHandoff"><option value="">— выберите —</option><option value="true">Да</option><option value="false">Нет</option></select></div>
        <div class="field full"><label for="knownContext">Что бот уже знает из предыдущих реплик <span class="hint">по одному пункту на строку; можно оставить пустым на первом ходе</span></label><textarea id="knownContext"></textarea></div>
      </div>
    </div>

    <div id="recordStatus" class="status">Статус: ожидает разметки</div>
    <div class="actions"><button id="pending">Сохранить черновик</button><button id="needsReview" class="warn">Передать эксперту</button><button id="approve" class="good">Утвердить и перейти дальше</button></div>
  </main>
</div>
<script>
const EMBEDDED_PACK = __PACK_JSON__;
const STORAGE_KEY = 'migtorg-stage5-labeling-' + EMBEDDED_PACK.dataset_version + '-' + EMBEDDED_PACK.source_sha256;
const byId = id => document.getElementById(id);
const clone = value => JSON.parse(JSON.stringify(value));
let pack = clone(EMBEDDED_PACK);
try { const saved = localStorage.getItem(STORAGE_KEY); if (saved) { const parsed=JSON.parse(saved); if (parsed.dataset_version===pack.dataset_version && parsed.source_sha256===pack.source_sha256) pack=parsed; } } catch (_) {}
let records = [];
let visible = [];
let position = 0;

function lines(value) { return String(value || '').split(/\r?\n/).map(x=>x.trim()).filter(Boolean); }
function boolValue(id) { const value=byId(id).value; return value==='' ? null : value==='true'; }
function flatten() {
  records = pack.cases.map(row=>({kind:'single', id:row.id, row, dialogue:null, turn:null}));
  for (const dialogue of pack.dialogues) for (const row of dialogue.turns) records.push({kind:'dialogue', id:row.id||dialogue.id+'-turn-'+String(row.turn).padStart(2,'0'), row, dialogue, turn:row.turn});
}
function statusOf(item) { return (item.row.review || {}).status || 'pending'; }
function applyFilter(keepId=true) {
  const current = visible[position]?.id;
  const filter=byId('filter').value;
  visible=records.filter(item => filter==='all' || filter===item.kind || (filter==='pending' && statusOf(item)!=='approved') || (filter==='needs_review' && statusOf(item)==='needs_review'));
  if (!visible.length) { visible=records; byId('filter').value='all'; }
  const found=keepId ? visible.findIndex(item=>item.id===current) : -1;
  position=found>=0 ? found : Math.min(position,visible.length-1);
  render();
}
function saveAttestation() {
  const a=pack.reviewer_attestation;
  a.reviewer_id=byId('reviewerId').value.trim();
  a.router_contributor=byId('independent').checked ? false : null;
  a.confidentiality_confirmed=byId('confidential').checked;
  a.personal_data_absent_confirmed=byId('privacy').checked;
}
function saveCurrent(status) {
  const item=visible[position]; if (!item) return;
  const expected={
    primary_topic:byId('primaryTopic').value.trim(), specific_situation:byId('specificSituation').value.trim(), bot_action:byId('botAction').value,
    required_information:byId('noRequired').checked ? [] : lines(byId('requiredInfo').value), no_required_information:byId('noRequired').checked,
    forbidden_information:byId('noForbidden').checked ? [] : lines(byId('forbiddenInfo').value), no_forbidden_information:byId('noForbidden').checked,
    multiple_valid_answers:byId('multipleValid').checked, acceptable_alternatives:byId('multipleValid').checked ? lines(byId('alternatives').value) : [], confidence:byId('confidence').value,
  };
  if (item.kind==='dialogue') Object.assign(expected,{continues_previous_topic:boolValue('continuesTopic'), known_context:lines(byId('knownContext').value), resolved_after_turn:boolValue('resolvedAfter'), support_handoff:boolValue('supportHandoff')});
  item.row.expected=expected;
  item.row.review={status:status || statusOf(item), reviewer_id:byId('reviewerId').value.trim(), router_contributor:byId('independent').checked ? false : null};
  saveAttestation(); persist(); updateProgress();
}
function validateItem(item) {
  const e=item.row.expected || {}; const errors=[];
  if (!String(e.primary_topic||'').trim()) errors.push('основная тема');
  if (!String(e.specific_situation||'').trim()) errors.push('конкретная ситуация');
  if (!['answer','clarify','support','out_of_scope','safe_refusal'].includes(e.bot_action)) errors.push('действие бота');
  if (!['high','medium'].includes(e.confidence)) errors.push('уверенность без экспертной проверки');
  if (!e.no_required_information && !(Array.isArray(e.required_information)&&e.required_information.length)) errors.push('обязательные сведения или отметка «нет»');
  if (!e.no_forbidden_information && !(Array.isArray(e.forbidden_information)&&e.forbidden_information.length)) errors.push('запрещённые сведения или отметка «нет»');
  if (e.multiple_valid_answers && !(Array.isArray(e.acceptable_alternatives)&&e.acceptable_alternatives.length)) errors.push('допустимые варианты');
  if (item.kind==='dialogue') for (const [key,label] of [['continues_previous_topic','продолжение темы'],['resolved_after_turn','решённость'],['support_handoff','передача сотруднику']]) if (typeof e[key] !== 'boolean') errors.push(label);
  if (!String((item.row.review||{}).reviewer_id||'').trim()) errors.push('идентификатор исполнителя');
  if ((item.row.review||{}).router_contributor !== false) errors.push('подтверждение независимости');
  return errors;
}
function persist() { try { localStorage.setItem(STORAGE_KEY,JSON.stringify(pack)); setGlobal('Черновик сохранён в этом браузере.','good'); } catch (_) { setGlobal('Не удалось сохранить в браузере — скачайте черновик JSON.','warn'); } }
function render() {
  const item=visible[position]; if (!item) return;
  const e=item.row.expected || {}, r=item.row.review || {};
  byId('recordMeta').replaceChildren(tag(item.id),tag(item.kind==='single'?'Одиночное обращение':'Диалог · ход '+item.turn),tag((position+1)+' из '+visible.length));
  byId('message').textContent=item.row.text;
  const context=byId('dialogueContext'); context.replaceChildren();
  if (item.kind==='dialogue') { context.classList.remove('hidden'); for (const turn of item.dialogue.turns) { const div=document.createElement('div'); div.className='turn'+(turn.turn===item.turn?' current':''); div.textContent='Ход '+turn.turn+': '+turn.text; context.appendChild(div); } } else context.classList.add('hidden');
  byId('dialogueFields').classList.toggle('hidden',item.kind!=='dialogue');
  byId('primaryTopic').value=e.primary_topic||''; byId('specificSituation').value=e.specific_situation||''; byId('botAction').value=e.bot_action||''; byId('confidence').value=e.confidence||'';
  byId('requiredInfo').value=(e.required_information||[]).join('\n'); byId('noRequired').checked=e.no_required_information===true;
  byId('forbiddenInfo').value=(e.forbidden_information||[]).join('\n'); byId('noForbidden').checked=e.no_forbidden_information===true;
  byId('multipleValid').checked=e.multiple_valid_answers===true; byId('alternatives').value=(e.acceptable_alternatives||[]).join('\n');
  byId('continuesTopic').value=typeof e.continues_previous_topic==='boolean'?String(e.continues_previous_topic):''; byId('knownContext').value=(e.known_context||[]).join('\n'); byId('resolvedAfter').value=typeof e.resolved_after_turn==='boolean'?String(e.resolved_after_turn):''; byId('supportHandoff').value=typeof e.support_handoff==='boolean'?String(e.support_handoff):'';
  toggleDisabled();
  const status=statusOf(item), box=byId('recordStatus'); box.className='status '+(status==='approved'?'good':status==='needs_review'?'warn':''); box.textContent='Статус: '+(status==='approved'?'утверждено':status==='needs_review'?'требуется эксперт':'ожидает разметки');
  byId('prev').disabled=position===0; byId('next').disabled=position===visible.length-1;
  updateProgress(); window.scrollTo({top:0,behavior:'smooth'});
}
function tag(text) { const span=document.createElement('span'); span.className='tag'; span.textContent=text; return span; }
function toggleDisabled() { byId('requiredInfo').disabled=byId('noRequired').checked; byId('forbiddenInfo').disabled=byId('noForbidden').checked; byId('alternatives').disabled=!byId('multipleValid').checked; }
function updateProgress() { const approved=records.filter(x=>statusOf(x)==='approved').length, expert=records.filter(x=>statusOf(x)==='needs_review').length; byId('progressBar').style.width=(approved/records.length*100)+'%'; byId('progressText').textContent=`Утверждено ${approved} из ${records.length}; эксперту: ${expert}`; }
function setGlobal(text,kind='') { const box=byId('globalStatus'); box.className='status '+kind; box.textContent=text; }
function download(finalMode) {
  saveCurrent(statusOf(visible[position])); saveAttestation(); const errors=[];
  if (finalMode) {
    if (!pack.reviewer_attestation.reviewer_id) errors.push('не указан исполнитель');
    if (pack.reviewer_attestation.router_contributor!==false) errors.push('не подтверждена независимость');
    if (!pack.reviewer_attestation.confidentiality_confirmed) errors.push('не подтверждена конфиденциальность');
    if (!pack.reviewer_attestation.personal_data_absent_confirmed) errors.push('не подтверждена проверка персональных данных');
    for (const item of records) { const itemErrors=validateItem(item); if (statusOf(item)!=='approved') itemErrors.push('статус не утверждён'); if (itemErrors.length) errors.push(item.id+': '+itemErrors.join(', ')); }
    if (errors.length) { setGlobal('Финальный экспорт заблокирован. '+errors.length+' замечаний. Первое: '+errors[0],'bad'); const firstId=errors[0].split(':')[0]; const idx=visible.findIndex(x=>x.id===firstId); if(idx>=0){position=idx;render();} return; }
    pack.reviewer_attestation.review_completed_at=new Date().toISOString();
  }
  const blob=new Blob([JSON.stringify(pack,null,2)+'\n'],{type:'application/json'}), url=URL.createObjectURL(blob), link=document.createElement('a'); link.href=url; link.download=finalMode?'stage5-blind-reviewed-pack.json':'stage5-blind-review-draft.json'; link.click(); setTimeout(()=>URL.revokeObjectURL(url),1000); setGlobal(finalMode?'Финальный JSON прошёл проверку и скачан.':'Черновик JSON скачан.','good');
}
function move(delta) { saveCurrent(statusOf(visible[position])); position=Math.max(0,Math.min(visible.length-1,position+delta)); render(); }

flatten();
byId('datasetMeta').textContent=pack.dataset_version+' · '+pack.cases.length+' одиночных · '+pack.dialogues.length+' диалогов';
byId('reviewerId').value=pack.reviewer_attestation.reviewer_id||''; byId('independent').checked=pack.reviewer_attestation.router_contributor===false; byId('confidential').checked=pack.reviewer_attestation.confidentiality_confirmed===true; byId('privacy').checked=pack.reviewer_attestation.personal_data_absent_confirmed===true;
byId('filter').addEventListener('change',()=>{saveCurrent(statusOf(visible[position]));applyFilter(false)}); byId('prev').onclick=()=>move(-1); byId('next').onclick=()=>move(1);
byId('pending').onclick=()=>{saveCurrent('pending');render()}; byId('needsReview').onclick=()=>{byId('confidence').value='requires_review';saveCurrent('needs_review');move(1)};
byId('approve').onclick=()=>{saveCurrent('approved');const errors=validateItem(visible[position]);if(errors.length){visible[position].row.review.status='pending';persist();setGlobal('Нельзя утвердить: '+errors.join(', ')+'.','bad');render();return;}move(1)};
byId('saveDraft').onclick=()=>download(false); byId('exportFinal').onclick=()=>download(true);
byId('noRequired').onchange=toggleDisabled; byId('noForbidden').onchange=toggleDisabled; byId('multipleValid').onchange=toggleDisabled;
for (const id of ['reviewerId','independent','confidential','privacy']) byId(id).addEventListener('change',()=>{saveAttestation();persist()});
byId('jump').addEventListener('change',()=>{const value=byId('jump').value.trim();const idx=visible.findIndex(x=>x.id===value||x.row.id===value);if(idx>=0){saveCurrent(statusOf(visible[position]));position=idx;render()}else setGlobal('ID не найден в текущем фильтре.','warn')});
byId('importFile').addEventListener('change',async event=>{try{const parsed=JSON.parse(await event.target.files[0].text());if(parsed.dataset_version!==EMBEDDED_PACK.dataset_version||parsed.source_sha256!==EMBEDDED_PACK.source_sha256)throw new Error('версия или источник не совпадают');pack=parsed;flatten();visible=records;position=0;persist();location.reload();}catch(error){setGlobal('Файл не загружен: '+error.message,'bad')}});
visible=records; render();
</script>
</body>
</html>'''


def build(pack: dict) -> str:
    embedded = json.dumps(pack, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return HTML.replace("__PACK_JSON__", embedded)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an offline independent-labeling form for stage 5.")
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    pack = json.loads(args.pack.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(build(pack).encode("utf-8"))
    print(json.dumps({
        "created": True,
        "output": str(args.output),
        "dataset_version": pack.get("dataset_version"),
        "single_turn_count": len(pack.get("cases", [])),
        "dialogue_count": len(pack.get("dialogues", [])),
        "dialogue_turn_count": sum(len(item.get("turns", [])) for item in pack.get("dialogues", [])),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
