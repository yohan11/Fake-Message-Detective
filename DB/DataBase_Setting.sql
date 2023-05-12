-- MySQL Workbench Forward Engineering

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- -----------------------------------------------------
-- Schema MESSAGE_DATA
-- -----------------------------------------------------

-- -----------------------------------------------------
-- Schema MESSAGE_DATA
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `MESSAGE_DATA` DEFAULT CHARACTER SET utf8 ;
USE `MESSAGE_DATA` ;

-- -----------------------------------------------------
-- Table `MESSAGE_DATA`.`user`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `MESSAGE_DATA`.`user` (
  `user_id` INT NOT NULL AUTO_INCREMENT,
  `user_name` VARCHAR(45) NULL,
  `user_phone` VARCHAR(45) NULL,
  PRIMARY KEY (`user_id`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `MESSAGE_DATA`.`message`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `MESSAGE_DATA`.`message` (
  `message_id` INT NOT NULL AUTO_INCREMENT,
  `user_id` INT NULL,
  `message_content` MEDIUMTEXT NULL,
  `message_sent_time` DATETIME NULL,
  `is_warning` TINYINT NULL,
  PRIMARY KEY (`message_id`),
  INDEX `fk_message_user_id_idx` (`user_id` ASC) VISIBLE,
  CONSTRAINT `fk_message_user_id`
    FOREIGN KEY (`user_id`)
    REFERENCES `MESSAGE_DATA`.`user` (`user_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `MESSAGE_DATA`.`warning`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `MESSAGE_DATA`.`warning` (
  `warning_id` INT NOT NULL AUTO_INCREMENT,
  `user_id` INT NULL,
  `message_id` INT NULL,
  `warning_time` DATETIME NULL,
  `warning_cause` MEDIUMTEXT NULL,
  `warning_valid` TINYINT NULL,
  PRIMARY KEY (`warning_id`),
  INDEX `fk_warning_user_id_idx` (`user_id` ASC) VISIBLE,
  INDEX `fk_warning_message_id_idx` (`message_id` ASC) VISIBLE,
  CONSTRAINT `fk_warning_user_id`
    FOREIGN KEY (`user_id`)
    REFERENCES `MESSAGE_DATA`.`user` (`user_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_warning_message_id`
    FOREIGN KEY (`message_id`)
    REFERENCES `MESSAGE_DATA`.`message` (`message_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
