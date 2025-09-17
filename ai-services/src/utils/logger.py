import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import json


class ColoredFormatter(logging.Formatter):
    """컬러가 적용된 로그 포맷터"""

    # ANSI 색상 코드
    COLORS = {
        'DEBUG': '\033[36m',    # 청록색
        'INFO': '\033[32m',     # 녹색
        'WARNING': '\033[33m',  # 노란색
        'ERROR': '\033[31m',    # 빨간색
        'CRITICAL': '\033[35m', # 마젠타
        'RESET': '\033[0m'      # 리셋
    }

    def format(self, record):
        # 색상 적용
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        # 레벨명에 색상 적용
        record.levelname = f"{color}{record.levelname}{reset}"

        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON 형식 로그 포맷터"""

    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # 예외 정보 추가
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        # 추가 필드 처리
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                         'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                         'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                         'thread', 'threadName', 'processName', 'process', 'message']:
                log_entry[key] = value

        return json.dumps(log_entry, ensure_ascii=False)


def setup_logger(name: str,
                level: str = "INFO",
                log_file: Optional[str] = None,
                console_output: bool = True,
                json_format: bool = False,
                max_file_size: int = 10 * 1024 * 1024,  # 10MB
                backup_count: int = 5) -> logging.Logger:
    """
    로거 설정

    Args:
        name: 로거 이름
        level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 로그 파일 경로
        console_output: 콘솔 출력 여부
        json_format: JSON 형식 사용 여부
        max_file_size: 로그 파일 최대 크기 (bytes)
        backup_count: 백업 파일 개수

    Returns:
        logging.Logger: 설정된 로거
    """
    logger = logging.getLogger(name)

    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 로그 레벨 설정
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # 포맷터 설정
    if json_format:
        formatter = JSONFormatter()
        console_formatter = JSONFormatter()
    else:
        # 상세한 포맷
        format_string = (
            '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s'
        )
        formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')

        # 콘솔용 컬러 포맷터
        console_formatter = ColoredFormatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')

    # 콘솔 핸들러 추가
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # 파일 핸들러 추가
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # 로테이팅 파일 핸들러
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 중복 로그 방지
    logger.propagate = False

    return logger


def setup_application_logger(app_name: str = "educational_ai",
                           config: Optional[dict] = None) -> logging.Logger:
    """
    애플리케이션 전체 로거 설정

    Args:
        app_name: 애플리케이션 이름
        config: 설정 딕셔너리

    Returns:
        logging.Logger: 메인 로거
    """
    if config is None:
        config = {}

    # 기본 설정
    log_level = config.get('log_level', 'INFO')
    log_file = config.get('log_file')
    debug = config.get('debug', False)
    verbose = config.get('verbose', False)

    # 로그 파일 경로 설정
    if log_file is None and debug:
        log_file = f"./logs/{app_name}_{datetime.now().strftime('%Y%m%d')}.log"

    # 메인 로거 설정
    main_logger = setup_logger(
        name=app_name,
        level=log_level,
        log_file=log_file,
        console_output=True,
        json_format=False
    )

    # 서브 모듈 로거들 설정
    module_loggers = [
        f"{app_name}.rag",
        f"{app_name}.models",
        f"{app_name}.utils"
    ]

    for module_name in module_loggers:
        module_logger = setup_logger(
            name=module_name,
            level=log_level if verbose else 'WARNING',
            log_file=log_file,
            console_output=verbose,
            json_format=False
        )

    # 외부 라이브러리 로거 레벨 조정
    external_loggers = [
        'openai',
        'httpx',
        'chromadb',
        'tiktoken'
    ]

    for ext_logger_name in external_loggers:
        ext_logger = logging.getLogger(ext_logger_name)
        ext_logger.setLevel(logging.WARNING if not debug else logging.INFO)

    main_logger.info(f"Logger initialized for {app_name}")
    return main_logger


def get_logger(name: str) -> logging.Logger:
    """
    이름으로 로거 가져오기

    Args:
        name: 로거 이름

    Returns:
        logging.Logger: 로거 인스턴스
    """
    return logging.getLogger(name)


def log_function_call(logger: logging.Logger):
    """
    함수 호출 로깅 데코레이터

    Args:
        logger: 사용할 로거

    Returns:
        decorator: 데코레이터 함수
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} failed with error: {str(e)}")
                raise
        return wrapper
    return decorator


def log_execution_time(logger: logging.Logger):
    """
    실행 시간 로깅 데코레이터

    Args:
        logger: 사용할 로거

    Returns:
        decorator: 데코레이터 함수
    """
    import time

    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {str(e)}")
                raise
        return wrapper
    return decorator


class LoggerMixin:
    """로거를 제공하는 믹스인 클래스"""

    @property
    def logger(self) -> logging.Logger:
        """클래스별 로거 반환"""
        return logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")


# 전역 로거 인스턴스
_application_logger: Optional[logging.Logger] = None


def get_application_logger() -> logging.Logger:
    """전역 애플리케이션 로거 반환"""
    global _application_logger
    if _application_logger is None:
        _application_logger = setup_application_logger()
    return _application_logger