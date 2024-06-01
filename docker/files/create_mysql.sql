create user 'jc'@'localhost' identified by 'jc@1234';
create database slurm_acct_db;
grant all PRIVILEGES on slurm_acct_db.* TO 'jc'@'localhost' with grant option;
