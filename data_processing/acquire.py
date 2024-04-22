import logging
import apartment
import solar_home
import l2rpn


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Getting the apartment data...")
    apartment.download()
    logging.info("Getting the solar home data...")
    solar_home.download()
    logging.info("Getting the l2rpn data...")
    l2rpn.download()
