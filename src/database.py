# src/database.py

from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config.settings import DB_URI

Base = declarative_base()

class Jogo(Base):
    __tablename__ = 'jogos'
    id = Column(Integer, primary_key=True)
    nome_time1 = Column(String)
    nome_time2 = Column(String)
    data = Column(String)
    
class Odds(Base):
    __tablename__ = 'odds'
    id = Column(Integer, primary_key=True)
    jogo_id = Column(Integer)
    mercado = Column(String)
    odds = Column(Float)

def criar_conexao():
    engine = create_engine(DB_URI)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()

def adicionar_jogo(nome_time1, nome_time2, data):
    session = criar_conexao()
    novo_jogo = Jogo(nome_time1=nome_time1, nome_time2=nome_time2, data=data)
    session.add(novo_jogo)
    session.commit()

def adicionar_odds(jogo_id, mercado, odds):
    session = criar_conexao()
    nova_odds = Odds(jogo_id=jogo_id, mercado=mercado, odds=odds)
    session.add(nova_odds)
    session.commit()
