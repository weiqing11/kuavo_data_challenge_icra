import logging
import sys
import os
import shutil
from tqdm.auto import tqdm

# =========================================================
# 1. 扩展 ANSI 颜色代码 (高颜值配色)
# =========================================================
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    TEXT_DEFAULT = "\033[39m"

    # 标准色
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m" # 紫色 (新增，很适合做边框)
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # 亮色 (High Intensity)
    BRIGHT_BLACK = "\033[90m" # 深灰
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

# =========================================================
# 2. Rank 过滤器 (保持逻辑不变)
# =========================================================
class RankFilter(logging.Filter):
    def filter(self, record):
        rank_keys = ["RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK"]
        current_rank = 0
        for key in rank_keys:
            val = os.environ.get(key)
            if val is not None:
                try:
                    current_rank = int(val)
                    break
                except ValueError:
                    continue
        if record.levelno >= logging.ERROR:
            return True
        return current_rank == 0

# =========================================================
# 3. 现代感 Formatter (Badge 风格)
# =========================================================
class ModernFormatter(logging.Formatter):
    """
    样式示例:
    10:23:45 | INFO | 🚀 Training started...
    """
    
    # 时间戳颜色
    TIME_CLR = Colors.BRIGHT_BLACK + Colors.ITALIC
    SEP_CLR = Colors.DIM + Colors.TEXT_DEFAULT
    MSG_CLR = Colors.TEXT_DEFAULT
    
    # 徽章样式定义
    LEVEL_BADGES = {
        logging.DEBUG:    f"{Colors.BOLD}{Colors.BLUE} DEBUG  {Colors.RESET}",
        logging.INFO:     f"{Colors.BOLD}{Colors.GREEN}  INFO  {Colors.RESET}",
        logging.WARNING:  f"{Colors.BOLD}{Colors.YELLOW}  WARN  {Colors.RESET}",
        logging.ERROR:    f"{Colors.BOLD}{Colors.RED} ERROR  {Colors.RESET}",
        logging.CRITICAL: f"{Colors.BOLD}{Colors.BRIGHT_RED} CRITIC {Colors.RESET}",
    }

    def format(self, record):
        # 1. 格式化时间
        asctime = self.formatTime(record, self.datefmt)
        time_str = f"{self.TIME_CLR}{asctime}{Colors.RESET}"
        
        # 2. 格式化级别
        level_badge = self.LEVEL_BADGES.get(record.levelno, record.levelname)
        
        # 3. 格式化分隔符
        sep = f"{self.SEP_CLR} | {Colors.RESET}"
        
        # 4. 组装消息
        msg = record.getMessage()
        
        if msg.startswith("\n"):
            # Box 模式：Header 打印完直接换行，Box 顶格显示
            return f"{time_str}{sep}{level_badge}{sep}{self.MSG_CLR}{msg}{Colors.RESET}"
        
        # 普通模式：如果是多行文本（比如报错 Traceback），保持缩进对齐比较好看
        elif "\n" in msg:
            indent = " " * 20 # 跟 Header 长度大致匹配
            msg = msg.replace("\n", "\n" + indent)
            
        return f"{time_str}{sep}{level_badge}{sep}{self.MSG_CLR}{msg}{Colors.RESET}"

# =========================================================
# 4. 初始化 Logger (霸道模式)
# =========================================================
def setup_logger(name="KuavoTrain", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False 
    if logger.hasHandlers():
        logger.handlers.clear()
    
    console_handler = logging.StreamHandler(sys.stdout)
    # 使用新的 Formatter，时间格式去掉毫秒更清爽
    console_handler.setFormatter(ModernFormatter(datefmt="%H:%M:%S")) 
    console_handler.addFilter(RankFilter())
    
    logger.addHandler(console_handler)
    return logger

logger = setup_logger()

# =========================================================
# 5. Box 打印工具 (圆角 + 优化排版)
# =========================================================
def log_box(title, content_dict, icon="🚀"):
    if not logger.isEnabledFor(logging.INFO): 
        return

    # 限制宽度，防止在超宽显示器上太丑
    term_width = min(shutil.get_terminal_size().columns, 90)
    
    # --- 样式定义 ---
    # 边框使用紫色，更有科技感
    C_BORDER = Colors.BRIGHT_MAGENTA 
    C_TITLE = Colors.BOLD + Colors.TEXT_DEFAULT
    C_KEY = Colors.CYAN  # 键使用青色
    C_VAL = Colors.TEXT_DEFAULT 
    C_ICON = Colors.RESET
    
    # 圆角字符
    TL, TR = "╭", "╮"
    BL, BR = "╰", "╯"
    H, V = "─", "│"
    
    lines = []
    
    # --- 1. 顶部标题栏 ---
    title_text = f" {icon}  {title} "
    # 计算标题长度 (去除颜色代码估算)
    content_len = len(title_text) + 2 # +2 是因为下面 padding 稍微留白
    
    padding_total = term_width - 2 - len(title_text)
    left_pad = padding_total // 2
    right_pad = padding_total - left_pad
    
    # 顶部线条： ╭──── Title ────╮
    lines.append(f"{C_BORDER}{TL}{H*left_pad}{C_TITLE}{title_text}{C_BORDER}{H*right_pad}{Colors.RESET}")
    
    # --- 2. 内容区域 ---
    if content_dict:
        max_key_len = max([len(str(k)) for k in content_dict.keys()])
        
        for i, (k, v) in enumerate(content_dict.items()):
            # 处理 Value 是列表的情况
            if isinstance(v, list):
                val_str = f"[{', '.join(str(x) for x in v)}]"
            else:
                val_str = str(v)
            
            # 键值对格式
            key_part = f"{C_KEY}{str(k).ljust(max_key_len)}{Colors.RESET}"
            val_part = f"{C_VAL}{val_str}{Colors.RESET}"
            
            # 组合一行： "│  Key   : Value   │"
            # 注意：手动计算可见字符长度用于 padding
            visible_len = 2 + max_key_len + 3 + len(str(val_str)) # 2(前缩进) + key + 3(" : ") + val
            space_padding = max(0, term_width - 2 - visible_len)
            
            # 使用点号虚线连接 Key 和 Value，增加可读性
            # 或者使用简单的冒号
            line_content = f"  {key_part} {Colors.DIM}:{Colors.RESET} {val_part}"
            
            lines.append(f"{C_BORDER}{V}{Colors.RESET}{line_content}{' ' * space_padding}{C_BORDER}{Colors.RESET}")
            
    # --- 3. 底部线条 ---
    lines.append(f"{C_BORDER}{BL}{H*(term_width-2)}{Colors.RESET}")
    
    # 为了防止多线程打印打断 Box，一次性输出
    logger.info("\n" + "\n".join(lines))

# =========================================================
# 6. 进度条打印工具
# =========================================================
class Progress:
    """
    统一风格的进度条，自动处理主进程判断和颜色格式化
    """
    def __init__(self, iterable, total=None, desc="Processing", disable=False):
        self.iterable = iterable
        self.total = total
        self.desc = desc
        self.disable = disable
        
        # 🎨 定制样式
        # l_bar: 左侧 (标题 + 百分比)
        # bar: 进度条本体
        # r_bar: 右侧 (计数 + 时间 + 参数)
        # 我们使用 ANSI 颜色注入到 bar_format 中
        
        # 进度条字符：使用平滑的块字符
        self.bar_format = (
            f"{Colors.BOLD}{Colors.CYAN}{{desc}}{Colors.RESET} "  # 标题 (青色粗体)
            f"{{percentage:3.0f}}% "                              # 百分比
            f"{Colors.MAGENTA}{{bar}}{Colors.RESET} "             # 进度条 (紫色)
            f"{{n_fmt}}/{{total_fmt}} "                           # 计数
            f"[{Colors.DIM}⏱️ {{elapsed}}<{{remaining}}{Colors.RESET}" # 时间 (灰色)
            f"{{postfix}}]"                                       # 后缀 (指标)
        )
        
        self.tqdm_instance = tqdm(
            iterable,
            total=total,
            desc=desc,
            disable=disable,
            bar_format=self.bar_format,
            ascii=" ▏▎▍▌▋▊▉█", # 使用平滑的 UTF-8 块字符
            leave=True,
            dynamic_ncols=True # 自动调整宽度
        )

    def __iter__(self):
        return iter(self.tqdm_instance)

    def update(self, n=1):
        self.tqdm_instance.update(n)

    def set_description(self, desc):
        self.tqdm_instance.set_description(desc)

    def set_postfix(self, **kwargs):
        """
        覆盖原版 set_postfix，自动把 key-value 格式化得更好看
        例如: loss=0.01 -> 📉 0.010
        """
        # 自定义映射图标
        icons = {
            "loss": "📉",
            "lr": "⚡",
            "acc": "🎯",
            "step": "👣"
        }
        
        formatted_list = []
        for k, v in kwargs.items():
            icon = icons.get(k, "")
            
            # 智能着色
            color = Colors.RESET
            if "loss" in k: color = Colors.BRIGHT_RED
            elif "lr" in k: color = Colors.BRIGHT_YELLOW
            elif "acc" in k: color = Colors.BRIGHT_GREEN
            
            # 格式化数值
            if isinstance(v, float):
                if v < 1e-4: val_str = f"{v:.1e}"
                else: val_str = f"{v:.4f}"
            else:
                val_str = str(v)
                
            formatted_list.append(f"{color}{icon} {val_str}{Colors.RESET}")
            
        # 使用 set_postfix_str 避免 tqdm 自动加 ", " 和 "="
        self.tqdm_instance.set_postfix_str("  ".join(formatted_list))

    def close(self):
        self.tqdm_instance.close()

# 方便调用的工厂函数
def get_progress_bar(iterable, desc, disable=False):
    return Progress(iterable, desc=desc, disable=disable)