import tarfile
import re
import glob
import smtplib
from urllib.request import urlopen
from html import unescape
from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr
from loguru import logger
import datetime
import time
from omegaconf import DictConfig

# NOTE:
# Keep PyMuPDF imports lazy so HTML-only workflows don't trigger MuPDF/cms warnings
# at module import time.

def extract_tex_code_from_tar(file_path:str, paper_id:str) -> dict[str,str]:
    try:
        tar = tarfile.open(file_path)
    except tarfile.ReadError:
        logger.debug(f"Failed to find main tex file of {paper_id}: Not a tar file.")
        return None
 
    tex_files = [f for f in tar.getnames() if f.endswith('.tex')]
    if len(tex_files) == 0:
        logger.debug(f"Failed to find main tex file of {paper_id}: No tex file.")
        tar.close()
        return None
    
    bbl_file = [f for f in tar.getnames() if f.endswith('.bbl')]
    match len(bbl_file) :
        case 0:
            if len(tex_files) > 1:
                logger.debug(f"Cannot find main tex file of {paper_id} from bbl: There are multiple tex files while no bbl file.")
                main_tex = None
            else:
                main_tex = tex_files[0]
        case 1:
            main_name = bbl_file[0].replace('.bbl','')
            main_tex = f"{main_name}.tex"
            if main_tex not in tex_files:
                logger.debug(f"Cannot find main tex file of {paper_id} from bbl: The bbl file does not match any tex file.")
                main_tex = None
        case _:
            logger.debug(f"Cannot find main tex file of {paper_id} from bbl: There are multiple bbl files.")
            main_tex = None

    if main_tex is None:
        logger.debug(f"Trying to choose tex file containing the document block as main tex file of {paper_id}")
    #read all tex files
    file_contents = {}
    for t in tex_files:
        f = tar.extractfile(t)
        content = f.read().decode('utf-8',errors='ignore')
        #remove comments
        content = re.sub(r'%.*\n', '\n', content)
        content = re.sub(r'\\begin{comment}.*?\\end{comment}', '', content, flags=re.DOTALL)
        content = re.sub(r'\\iffalse.*?\\fi', '', content, flags=re.DOTALL)
        #remove redundant \n
        content = re.sub(r'\n+', '\n', content)
        content = re.sub(r'\\\\', '', content)
        #remove consecutive spaces
        content = re.sub(r'[ \t\r\f]{3,}', ' ', content)
        if main_tex is None and re.search(r'\\begin\{document\}', content):
            main_tex = t
            logger.debug(f"Choose {t} as main tex file of {paper_id}")
        file_contents[t] = content
    
    if main_tex is not None:
        main_source:str = file_contents[main_tex]
        #find and replace all included sub-files
        include_files = re.findall(r'\\input\{(.+?)\}', main_source) + re.findall(r'\\include\{(.+?)\}', main_source)
        for f in include_files:
            if not f.endswith('.tex'):
                file_name = f + '.tex'
            else:
                file_name = f
            main_source = main_source.replace(f'\\input{{{f}}}', file_contents.get(file_name, ''))
        file_contents["all"] = main_source
    else:
        logger.debug(f"Failed to find main tex file of {paper_id}: No tex file containing the document block.")
        file_contents["all"] = None
        
    tar.close()
    return file_contents

def extract_markdown_from_pdf(file_path:str) -> str:
    try:
        import pymupdf.layout as _pymupdf_layout
        _pymupdf_layout.activate()
    except Exception as e:
        logger.warning(f"Failed to activate pymupdf.layout, continue without layout mode: {e}")

    try:
        import pymupdf4llm
    except Exception as e:
        raise RuntimeError(f"pymupdf4llm is unavailable: {e}")

    return pymupdf4llm.to_markdown(file_path,use_ocr=False,header=False,footer=False,ignore_code=True)

def extract_text_from_arxiv_html(html_url:str) -> str:
    with urlopen(html_url, timeout=20) as resp:
        html = resp.read().decode("utf-8", errors="ignore")

    # Remove scripts/styles and all tags, keep plain text.
    html = re.sub(r"<script[\\s\\S]*?</script>", " ", html, flags=re.IGNORECASE)
    html = re.sub(r"<style[\\s\\S]*?</style>", " ", html, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    # Cap to avoid huge prompts downstream.
    return text[:50000]

def glob_match(path:str, pattern:str) -> bool:
    re_pattern = glob.translate(pattern,recursive=True)
    return re.match(re_pattern, path) is not None

def _build_smtp_server(smtp_server:str, smtp_port:int):
    try:
        server = smtplib.SMTP(smtp_server, smtp_port, timeout=20)
        server.starttls()
        return server
    except Exception as e:
        logger.debug(f"Failed to use TLS. {e} | try SSL")
    try:
        return smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=20)
    except Exception as e:
        logger.debug(f"Failed to use SSL. {e} | try plain SMTP")
    return smtplib.SMTP(smtp_server, smtp_port, timeout=20)


def send_email(config:DictConfig, html:str):
    sender = config.email.sender
    receiver = config.email.receiver
    password = config.email.sender_password
    smtp_server = config.email.smtp_server
    smtp_port = config.email.smtp_port
    retry = int(config.email.get("retry_times", 3))
    retry_interval = float(config.email.get("retry_interval_sec", 3.0))

    def _format_addr(s):
        name, addr = parseaddr(s)
        return formataddr((Header(name, 'utf-8').encode(), addr))

    msg = MIMEText(html, 'html', 'utf-8')
    msg['From'] = _format_addr('Github Action <%s>' % sender)
    msg['To'] = _format_addr('You <%s>' % receiver)
    today = datetime.datetime.now().strftime('%Y/%m/%d')
    msg['Subject'] = Header(f'Daily arXiv {today}', 'utf-8').encode()

    last_err = None
    for i in range(1, retry + 1):
        server = None
        try:
            server = _build_smtp_server(smtp_server, smtp_port)
            server.login(sender, password)
            server.sendmail(sender, [receiver], msg.as_string())
            logger.info(f"Email sent successfully on attempt {i}/{retry}")
            return
        except Exception as e:
            last_err = e
            logger.warning(f"Send email failed on attempt {i}/{retry}: {e}")
            if i < retry:
                time.sleep(retry_interval)
        finally:
            if server is not None:
                try:
                    server.quit()
                except Exception:
                    pass

    raise RuntimeError(f"Failed to send email after {retry} attempts: {last_err}")