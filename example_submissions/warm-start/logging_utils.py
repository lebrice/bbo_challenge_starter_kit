import logging
import sys
from pathlib import Path

logging.basicConfig(
    format='%(levelname)-8s [%(name)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO,
)
logging.getLogger("GP").setLevel(logging.WARNING)
logging.getLogger("BayesianGPLVM").setLevel(logging.WARNING)
logging.getLogger("sparse gp").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
        """ Gets a logger for the given file. Sets a nice default format. 
        TODO: figure out if we should add handlers, etc. 
        """
        try:
            p = Path(name)
            if p.exists():
                name = str(p.absolute().relative_to(Path.cwd()).as_posix())
        except:
            pass
        logger = logging.getLogger(name)
        if "-d" in sys.argv:
            logger.setLevel(logging.DEBUG)
        # logger.addHandler(TqdmLoggingHandler())
        return logger
