"""
FIM (Fill-In-the-Middle) 代码补全模型评估脚本 (DeepSeek 版本)

功能概述
--------
基于 HuggingFace Transformers 对一个或多个 checkpoint 进行批量推理与质量评估,
支持多 GPU、每卡多 worker 的进程级并行, 并按统一口径计算多种文本相似度指标。
本脚本针对 DeepSeek 系列模型, 使用 DeepSeek 的 FIM 占位符与停止词。

主要能力
--------
1. 数据加载
   - 支持 .json / .parquet 两种测试集格式
   - parquet 自动从 reward_model.ground_truth 取 gt
   - 使用 tokenizer 过滤超过 15k token 的样本

2. 并行推理
   - 通过 multiprocessing(spawn) 启动若干 worker 进程
   - 每个 worker 绑定到指定 GPU, 加载一份模型
   - 主进程通过 task_queue / result_queue 分发与回收任务
   - 推理结果按原始顺序还原, 同时以 jsonl 增量落盘
   - 使用 DeepSeek FIM 格式: <｜fim▁begin｜>prefix<｜fim▁hole｜>suffix<｜fim▁end｜>
   - 输入若已是 Qwen FIM 占位符, 会自动映射为 DeepSeek FIM 占位符

3. 评估指标 (gt 与 pred 均做 strip; 不做长度截断, 整串比较)
   - es     : Edit Similarity = 1 - 归一化 Levenshtein 距离
   - em     : Exact-Match (严格全等)
   - rougeL : ROUGE-L F1
   - jw     : Jaro-Winkler 相似度
   - avg    : (es + em + rougeL) / 3
   - avg_jw : (es + em + rougeL + jw) / 4
   空值约定: 双方都为空 -> 1.0, 仅一方为空 -> 0.0

4. 结果分组与报告
   - 按 ground truth 是否为空将样本分为两组分别统计
   - 计算两类条件概率:
       * gt 为空 且 预测为空 的概率
       * gt 非空 且 预测非空 的概率
   - 输出: 逐条结果 json、summary json、人类可读 metric.txt、汇总 markdown

输出目录
--------
固定写入脚本同目录下:
  - prediction/<model_alia>_epoch_<n>_result/   逐 epoch 推理与评估结果
  - record/<file>_resultrecord_<alia>_<data_name>.md   汇总 markdown

命令行参数
----------
  --model_path        模型路径 (单 checkpoint, 或配合 --is_parent 指向父目录)
  --is_parent         model_path 是否为包含多个 checkpoint-* 子目录的父目录
  --model_alia        模型别名 (用于结果命名)
  --data_path         测试集路径 (.json 或 .parquet)
  --data_name         测试集名称 (用于结果命名)
  --gpu_ids           可用 GPU 列表, 逗号分隔, 如 "4,5,6,7"
  --workers_per_gpu   每张 GPU 启动的 worker 数 (默认 2)
"""
import pandas as pd
import torch
import json
from rouge_score import rouge_scorer
import Levenshtein
from tqdm import tqdm
import os
import argparse
import multiprocessing as mp
import time

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


# ---------------------------------------------------------------------------
# 评分前置: 对 gt / pred 做 strip 与空值判定
# ---------------------------------------------------------------------------
def _prepare(gt, pred):
    """
    返回 (gt_s, pred_s, edge_score):
      - edge_score 为 float -> 直接作为最终分数返回 (二者都空=1, 仅一方空=0)
      - edge_score 为 None  -> 二者都非空, 由调用方对整串 gt_s / pred_s 做指标计算
    """
    gt_s = str(gt).strip()
    pred_s = str(pred).strip()
    g_empty, p_empty = (len(gt_s) == 0), (len(pred_s) == 0)
    if g_empty and p_empty:
        return gt_s, pred_s, 1.0
    if g_empty or p_empty:
        return gt_s, pred_s, 0.0
    return gt_s, pred_s, None


# ---------------------------------------------------------------------------
# 文本相似度指标
# ---------------------------------------------------------------------------
def cal_es(gt, pred):
    """Edit Similarity = 1 - 归一化 Levenshtein 距离 (整串)"""
    gt, pred, edge = _prepare(gt, pred)
    if edge is not None:
        return edge
    return 1.0 - Levenshtein.distance(gt, pred) / max(len(gt), len(pred))


def cal_em(gt, pred):
    """Exact-Match (严格全等): gt 与 pred strip 后完全相同则为 1"""
    gt, pred, edge = _prepare(gt, pred)
    if edge is not None:
        return edge
    return 1.0 if gt == pred else 0.0


def cal_rouge(gt, pred):
    """ROUGE-L F1 (整串)"""
    gt, pred, edge = _prepare(gt, pred)
    if edge is not None:
        return edge
    return scorer.score(gt, pred)["rougeL"].fmeasure


def _jaro_winkler_core(s1, s2):
    """标准 Jaro-Winkler 相似度计算 (假定输入均非空)."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    max_len = max(len1, len2)
    match_window = max(max_len // 2 - 1, 0)
    match1 = [False] * len1
    match2 = [False] * len2
    matches = 0
    for i in range(len1):
        start = max(0, i - match_window)
        end = min(i + match_window + 1, len2)
        for j in range(start, end):
            if not match2[j] and s1[i] == s2[j]:
                match1[i] = True
                match2[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0
    transpositions = 0
    j = 0
    for i in range(len1):
        if match1[i]:
            while not match2[j]:
                j += 1
            if s1[i] != s2[j]:
                transpositions += 1
            j += 1
    transpositions //= 2
    jaro = (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3
    prefix_len = 0
    max_prefix = min(4, len1, len2)
    for i in range(max_prefix):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break
    return jaro + prefix_len * 0.1 * (1 - jaro)


def cal_jw(gt, pred):
    """Jaro-Winkler 相似度 (整串)"""
    gt, pred, edge = _prepare(gt, pred)
    if edge is not None:
        return edge
    return _jaro_winkler_core(gt, pred)


# ---------------------------------------------------------------------------
# 空值判定 (用于按 gt 分组及条件正确率统计)
# ---------------------------------------------------------------------------
def is_pred_empty(pred):
    """判定预测结果 strip 后是否为空."""
    return len(str(pred).strip()) == 0


def is_gt_empty(gt):
    """判定 ground truth strip 后是否为空."""
    return len(str(gt).strip()) == 0


# ---------------------------------------------------------------------------
# 指标聚合
# ---------------------------------------------------------------------------
def compute_metrics(records):
    if len(records) == 0:
        empty_metrics = {
            'es': None, 'em': None, 'rougel': None, 'jw': None,
            'avg': None, 'avg_jw': None,
            'count': 0, 'pred_nonempty': 0, 'pred_empty': 0,
        }
        return pd.DataFrame(records), empty_metrics

    df = pd.DataFrame(records)
    # 对每行样本逐一计算 4 个相似度指标
    df['es']     = df.apply(lambda r: cal_es   (r['user_gt'], r['test_answer']), axis=1)
    df['em']     = df.apply(lambda r: cal_em   (r['user_gt'], r['test_answer']), axis=1)
    df['rougel'] = df.apply(lambda r: cal_rouge(r['user_gt'], r['test_answer']), axis=1)
    df['jw']     = df.apply(lambda r: cal_jw   (r['user_gt'], r['test_answer']), axis=1)
    df['avg']    = (df['es'] + df['em'] + df['rougel']) / 3
    df['avg_jw'] = (df['es'] + df['em'] + df['rougel'] + df['jw']) / 4

    df['gt_is_empty']   = df['user_gt'].apply(is_gt_empty)
    df['pred_is_empty'] = df['test_answer'].apply(is_pred_empty)

    metrics = {
        'es':            df['es'].mean(),
        'em':            df['em'].mean(),
        'rougel':        df['rougel'].mean(),
        'jw':            df['jw'].mean(),
        'avg':           df['avg'].mean(),
        'avg_jw':        df['avg_jw'].mean(),
        'count':         len(df),
        'pred_nonempty': int((~df['pred_is_empty']).sum()),
        'pred_empty':    int(df['pred_is_empty'].sum()),
    }
    return df, metrics


def fmt(v, nd=4):
    if v is None:
        return 'NA'
    return f'{v:.{nd}f}'


# ---------------------------------------------------------------------------
# 推理 Worker: 每个进程绑定一张 GPU, 加载一份模型, 循环从 task_queue 取任务推理
# ---------------------------------------------------------------------------
def _worker_loop(worker_id, gpu_id, model_path, task_queue, result_queue, ready_queue):
    try:
        torch.cuda.set_device(int(gpu_id))
    except Exception as e:
        print(f"[worker {worker_id}] set_device({gpu_id}) failed: {e}", flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = torch.device(f'cuda:{gpu_id}')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    generate_kwargs = {
        "max_new_tokens": 60,
        "temperature": 0.01,
        "top_p": 0.7,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        'use_cache': True,
    }
    # DeepSeek 系列模型的特殊 token 与停止词
    stop_sequences = ["<｜fim▁hole｜>", "<｜fim▁begin｜>", "<｜fim▁end｜>",
                      "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>",
                      "<pad>", "<|User|>", "<|Assistant|>", "<|EOT|>", "\n"]
    eos_ids = []
    for stop in stop_sequences:
        try:
            tok = tokenizer.encode(stop, add_special_tokens=False)
            if len(tok) > 0:
                eos_ids.append(tok[0])
        except Exception:
            pass
    eos_ids.append(tokenizer.eos_token_id)
    generate_kwargs["eos_token_id"] = eos_ids

    ready_queue.put(worker_id)
    print(f"[worker {worker_id}] ready on cuda:{gpu_id}, model={model_path}", flush=True)

    while True:
        item = task_queue.get()
        if item is None:
            print(f"[worker {worker_id}] received stop signal, exit.", flush=True)
            break
        idx, line = item
        try:
            if 'prompt' in line:
                input_str = line['prompt']
                # 若 prompt 使用 Qwen FIM 占位符, 映射为 DeepSeek FIM 占位符
                input_str = input_str.replace('<|fim_prefix|>', '<｜fim▁begin｜>')
                input_str = input_str.replace('<|fim_suffix|>', '<｜fim▁hole｜>')
                input_str = input_str.replace('<|fim_middle|>', '<｜fim▁end｜>')
            else:
                # 使用 DeepSeek FIM 格式拼接 prefix / suffix
                input_str = f"<｜fim▁begin｜>{line['prefix']}<｜fim▁hole｜>{line['suffix']}<｜fim▁end｜>"
            input_ids = tokenizer.encode(input_str, return_tensors="pt").to(device)
            with torch.no_grad():
                response = model.generate(input_ids=input_ids, **generate_kwargs)
                response = tokenizer.decode(response[0][len(input_ids[0]):], skip_special_tokens=True)
            for stop_ in stop_sequences:
                if stop_ in response:
                    response = response.split(stop_)[0]
            line['test_answer'] = response
        except Exception as e:  # pylint: disable=broad-except
            line['test_answer'] = ''
            line['_error'] = f'{type(e).__name__}: {e}'
            print(f"[worker {worker_id}] infer error on idx={idx}: {e}", flush=True)
        result_queue.put((idx, line))


def run_parallel_inference(model_path, to_predict_data, gpu_ids, workers_per_gpu, infer_results_path):
    """启动一组 worker 进行多 GPU 并行推理, 按原始顺序汇总并落盘所有结果."""
    worker_specs = []
    wid = 0
    for g in gpu_ids:
        for _ in range(workers_per_gpu):
            worker_specs.append((wid, g))
            wid += 1
    n_workers = len(worker_specs)
    print(f"[main] launching {n_workers} workers across gpus={gpu_ids} "
          f"(workers_per_gpu={workers_per_gpu})", flush=True)

    ctx = mp.get_context('spawn')
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    ready_queue = ctx.Queue()

    procs = []
    for worker_id, gpu_id in worker_specs:
        p = ctx.Process(
            target=_worker_loop,
            args=(worker_id, gpu_id, model_path, task_queue, result_queue, ready_queue),
            daemon=False,
        )
        p.start()
        procs.append(p)

    ready_count = 0
    while ready_count < n_workers:
        ready_queue.get()
        ready_count += 1
        print(f"[main] worker ready: {ready_count}/{n_workers}", flush=True)

    total = len(to_predict_data)
    for idx, line in enumerate(to_predict_data):
        task_queue.put((idx, line))
    for _ in range(n_workers):
        task_queue.put(None)

    results = [None] * total
    with open(infer_results_path, 'w', encoding='utf-8') as infer_f:
        with tqdm(total=total, desc='inference') as pbar:
            received = 0
            while received < total:
                idx, line = result_queue.get()
                results[idx] = line
                infer_f.write(json.dumps(line, ensure_ascii=False) + '\n')
                infer_f.flush()
                received += 1
                pbar.update(1)

    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            print(f"[main] worker pid={p.pid} still alive, terminate.", flush=True)
            p.terminate()
            p.join()
    return results


# ---------------------------------------------------------------------------
# 评估与报告: 输出 overall / gt 为空 / gt 非空 三组指标及两个条件正确率
# ---------------------------------------------------------------------------
def evaluate_and_report(all_res, model_item_path, args, epoch, output_dir):
    data_all, m_all = compute_metrics(all_res)

    gt_empty_records    = [r for r in all_res if is_gt_empty(r['user_gt'])]
    gt_nonempty_records = [r for r in all_res if not is_gt_empty(r['user_gt'])]
    _, m_gt_empty    = compute_metrics(gt_empty_records)
    _, m_gt_nonempty = compute_metrics(gt_nonempty_records)

    n_gt_empty = len(gt_empty_records)
    n_gt_nonempty = len(gt_nonempty_records)
    n_gt_empty_pred_empty       = sum(1 for r in gt_empty_records    if is_pred_empty(r['test_answer']))
    n_gt_nonempty_pred_nonempty = sum(1 for r in gt_nonempty_records if not is_pred_empty(r['test_answer']))
    p_empty_correct    = (n_gt_empty_pred_empty       / n_gt_empty)    if n_gt_empty    > 0 else None
    p_nonempty_correct = (n_gt_nonempty_pred_nonempty / n_gt_nonempty) if n_gt_nonempty > 0 else None

    # 写出逐条评估结果与 summary
    with open(output_dir + f"/{args.data_name}_eval_results.json", "w", encoding='utf-8') as f:
        json.dump(data_all.to_dict("records"), f, indent=4, ensure_ascii=False)
    summary = {
        'model': model_item_path,
        'data_name': args.data_name,
        'epoch': epoch,
        'overall': m_all,
        'gt_empty': m_gt_empty,
        'gt_nonempty': m_gt_nonempty,
        'prob_gt_empty_pred_empty': p_empty_correct,
        'prob_gt_nonempty_pred_nonempty': p_nonempty_correct,
        'n_gt_empty': n_gt_empty,
        'n_gt_nonempty': n_gt_nonempty,
    }
    with open(output_dir + f"/{args.data_name}_summary.json", "w", encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    # 输出人类可读的 metric.txt 与 stdout
    lines = [
        f'the model {model_item_path} on Test-data {args.data_name} is',
        f'after predict len is  {len(all_res)}',
        '',
        '==== 完整测试集 overall ====',
        f"es(1-编辑距离): {fmt(m_all['es'])}",
        f"em平均分(严格全等): {fmt(m_all['em'])}",
        f"rouge-l: {fmt(m_all['rougel'])}",
        f"jw: {fmt(m_all['jw'])}",
        f"平均分:   {fmt(m_all['avg'])}",
        f"平均分(jw):   {fmt(m_all['avg_jw'])}",
        f"总数量:{m_all['count']}, 预测非空数量：{m_all['pred_nonempty']}",
        '',
        '==== gt为空 ====',
        f"样本数:{m_gt_empty['count']}, es:{fmt(m_gt_empty['es'])}, em:{fmt(m_gt_empty['em'])}, "
        f"rouge-l:{fmt(m_gt_empty['rougel'])}, avg:{fmt(m_gt_empty['avg'])}, jw:{fmt(m_gt_empty['jw'])}",
        f"gt为空且模型正确预测为空的概率: {fmt(p_empty_correct)}  ({n_gt_empty_pred_empty}/{n_gt_empty})",
        '',
        '==== gt非空 ====',
        f"样本数:{m_gt_nonempty['count']}, es:{fmt(m_gt_nonempty['es'])}, em:{fmt(m_gt_nonempty['em'])}, "
        f"rouge-l:{fmt(m_gt_nonempty['rougel'])}, avg:{fmt(m_gt_nonempty['avg'])}, jw:{fmt(m_gt_nonempty['jw'])}",
        f"gt非空且模型正确预测非空的概率: {fmt(p_nonempty_correct)}  ({n_gt_nonempty_pred_nonempty}/{n_gt_nonempty})",
        '===' * 50,
        '',
    ]
    for l in lines:
        print(l)
    with open(output_dir + "/metric.txt", 'a') as fw:
        fw.write('\n'.join(lines) + '\n')
        fw.write('预测结束\n\n')

    return m_all, m_gt_empty, m_gt_nonempty, n_gt_empty, n_gt_nonempty, p_empty_correct, p_nonempty_correct


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='待评估模型的父目录')
    parser.add_argument('--is_parent', default=False, action='store_true')
    parser.add_argument('--model_alia', type=str, required=True, help='模型别名，当有多个训练模型时用作区分')
    parser.add_argument('--data_path', type=str, required=True, help='测试集路径')
    parser.add_argument('--data_name', type=str, required=True, help='测试集名称')
    parser.add_argument('--gpu_ids', type=str, default='4,5,6,7',
                        help='可用 GPU 列表, 逗号分隔, 例如 "4,5,6,7"')
    parser.add_argument('--workers_per_gpu', type=int, default=2,
                        help='每张 GPU 启动的推理 worker 数 (默认 2)')
    args = parser.parse_args()

    # 结果固定写入脚本同目录下的 prediction/ 和 record/
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PREDICTION_DIR = os.path.join(SCRIPT_DIR, 'prediction')
    RECORD_DIR = os.path.join(SCRIPT_DIR, 'record')
    os.makedirs(PREDICTION_DIR, exist_ok=True)
    os.makedirs(RECORD_DIR, exist_ok=True)

    gpu_ids = [g.strip() for g in args.gpu_ids.split(',') if g.strip() != '']
    assert len(gpu_ids) > 0, 'gpu_ids 不能为空'
    assert args.workers_per_gpu >= 1, 'workers_per_gpu 必须 >= 1'

    # ---------- 加载数据 ----------
    if args.data_path.endswith(".json"):
        with open(args.data_path, 'r') as f:
            to_predict_data = json.load(f)
    elif args.data_path.endswith(".parquet"):
        data_ = pd.read_parquet(args.data_path)
        data_ = [data_.iloc[i].to_dict() for i in range(len(data_))]
        to_predict_data = [{'prompt': obj['prompt'],
                            'user_gt': obj['reward_model']['ground_truth']} for obj in data_]
    else:
        raise ValueError(f'Unsupported data file: {args.data_path}')
    print("dataset example:", to_predict_data[0])

    # ---------- 收集 epoch 列表 ----------
    if args.is_parent:
        all_epochs = []
        for item in os.listdir(args.model_path):
            item_path = os.path.join(args.model_path, item)
            if os.path.isdir(item_path) and item.startswith('checkpoint'):
                all_epochs.append(item_path)
        assert all_epochs, 'Not found checkpoints'
        all_epochs.sort(key=lambda s: int(s.split('checkpoint-')[1]))
    else:
        all_epochs = [args.model_path]

    # ---------- 用首个 checkpoint 的 tokenizer 过滤过长样本 ----------
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(all_epochs[0])
    print("初始条数:", len(to_predict_data))
    try:
        to_predict_data = [obj for obj in to_predict_data
                           if len(_tok.encode(obj['prompt'])) < 15360]
        print("过滤超过15k的items,剩余:", len(to_predict_data))
    except Exception as e:
        print(e)
    del _tok

    # ---------- 准备 markdown 汇总表 ----------
    markdown_str = '''### 完整测试集 (overall, 在全部样本上计算)
|模型|epoch|data_name|es|em|rouge-l|avg|总数|预测非空数|jw|avg_jw|
|----|----|----|----|----|----|----|----|----|----|----|
'''
    markdown_split = '''### 按 ground truth 空 / 非空 分组
|模型|epoch|data_name|分组|es|em|rouge-l|avg|jw|avg_jw|该组样本数|预测非空数|预测空数|
|----|----|----|----|----|----|----|----|----|----|----|----|----|
'''
    markdown_prob = '''### 空 / 非空 预测正确率
|模型|epoch|data_name|gt为空样本数|gt为空且预测为空的概率|gt非空样本数|gt非空且预测非空的概率|
|----|----|----|----|----|----|----|----|
'''

    # ---------- 逐 epoch 推理与评估 ----------
    for epoch_idx, model_item_path in enumerate(all_epochs):
        epoch = epoch_idx + 1
        output_dir = os.path.join(PREDICTION_DIR, f'{args.model_alia}_epoch_{epoch}_result')
        os.makedirs(output_dir, exist_ok=True)
        infer_results_path = output_dir + f"/{args.data_name}_infer_results.jsonl"

        t0 = time.time()
        all_res = run_parallel_inference(
            model_path=model_item_path,
            to_predict_data=to_predict_data,
            gpu_ids=gpu_ids,
            workers_per_gpu=args.workers_per_gpu,
            infer_results_path=infer_results_path,
        )
        all_res = [r for r in all_res if r is not None]
        print(f"[main] epoch {epoch} inference done, "
              f"items={len(all_res)}, cost={time.time() - t0:.1f}s", flush=True)

        (m_all, m_gt_empty, m_gt_nonempty,
         n_gt_empty, n_gt_nonempty,
         p_empty_correct, p_nonempty_correct) = evaluate_and_report(
            all_res, model_item_path, args, epoch, output_dir,
        )

        markdown_str += (
            f"|{args.model_alia}|{epoch}|{args.data_name}|"
            f"{fmt(m_all['es'])}|{fmt(m_all['em'])}|{fmt(m_all['rougel'])}|{fmt(m_all['avg'])}|"
            f"{m_all['count']}|{m_all['pred_nonempty']}|{fmt(m_all['jw'])}|{fmt(m_all['avg_jw'])}|\n"
        )
        for tag, mm in [('gt为空', m_gt_empty), ('gt非空', m_gt_nonempty)]:
            markdown_split += (
                f"|{args.model_alia}|{epoch}|{args.data_name}|{tag}|"
                f"{fmt(mm['es'])}|{fmt(mm['em'])}|{fmt(mm['rougel'])}|{fmt(mm['avg'])}|"
                f"{fmt(mm['jw'])}|{fmt(mm['avg_jw'])}|{mm['count']}|{mm['pred_nonempty']}|{mm['pred_empty']}|\n"
            )
        markdown_prob += (
            f"|{args.model_alia}|{epoch}|{args.data_name}|"
            f"{n_gt_empty}|{fmt(p_empty_correct)}|{n_gt_nonempty}|{fmt(p_nonempty_correct)}|\n"
        )

    full_markdown = markdown_str + '\n' + markdown_split + '\n' + markdown_prob
    print(f'MarkDown str is\n{full_markdown}')
    file_name = os.path.basename(__file__)
    ali_name = args.model_alia.split('_split_')[0]
    save_file = os.path.join(RECORD_DIR, f"{file_name}_resultrecord_{ali_name}_{args.data_name}.md")
    with open(save_file, 'a') as writer:
        writer.write('\n' + full_markdown + '\n')
