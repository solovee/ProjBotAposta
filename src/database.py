from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from config.settings import DB_URI

Base = declarative_base()

engine = create_engine(DB_URI)
SessionLocal = scoped_session(sessionmaker(bind=engine))

def get_session():
    return SessionLocal()

class Jogo(Base):
    __tablename__ = 'jogos'
    id = Column(Integer, primary_key=True)
    nome_time1 = Column(String, nullable=False)
    nome_time2 = Column(String, nullable=False)
    data = Column(String, nullable=False)
    tipo_linha = Column(String, nullable=False)  # Ex: "Under" ou "Over"
    odd = Column(Float, nullable=False)  # Ex: 2.1
    gols_linha = Column(Float, nullable=False)  # Ex: 2.0
    media_gols_confronto = Column(Float, nullable=False)  # Média esperada

Base.metadata.create_all(engine)

def salvar_linhas_desreguladas(linhas_desreguladas):
    """Salva apenas as linhas desreguladas no banco."""
    session = get_session()
    try:
        for linha in linhas_desreguladas:
            novo_jogo = Jogo(
                id=linha["game_id"],
                nome_time1=linha["home_team"],
                nome_time2=linha["away_team"],
                data=linha["data"],
                tipo_linha=linha["tipo"],
                odd=linha["odds"],
                gols_linha=linha["linha"],
                media_gols_confronto=linha["media_esperada"]
            )
            session.add(novo_jogo)
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Erro ao salvar linhas desreguladas: {e}")
    finally:
        session.close()
