import time as time_module 
from datetime import datetime, timedelta, time



from api import BetsAPIClient, dia_anterior
import pandas as pd
from dotenv import load_dotenv
import os
import threading
import NN
import telegramBot as tb
import logging
import json
import threading
import os
import signal
import sys
import mlp_pois
import database




# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)



load_dotenv()

api = os.getenv("API_KEY")
chat_id = int(os.getenv("CHAT_ID"))
#novo -4954876315
# -1002610837223
chats = [chat_id, -4954876315]
chats_all = [chat_id, -4954876315, -1002610837223]
resultados_medias = {8: 2.49, 12: 2.18}


apiclient = BetsAPIClient(api_key=api)



CSV_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resultados_60.csv')
#lista dos thresholds das nns
lista_th = [0.6,0.35,0.65,0.6,0.6,0.65]
list_checa = []
list_uni = [0]




#!pega uma vez ao virar o dia e depois de 20 em 20 min pra testar
data_hoje = datetime.now().date().strftime('%Y%m%d')
#!reseta todo dia as 00:00, guarda jogos programados do dia, deve zerar ao mudar o dia
programado = []

def checa_virada_do_dia():
    '''reseta os jogos programados do dia'''
    global data_hoje, programado
    while True:
        novo_dia = datetime.now().date().strftime('%Y%m%d')
        if novo_dia != data_hoje:
            data_hoje = novo_dia
            programado = []
            database.insert_csv_to_db(CSV_FILE, "dados_resultados")
            logger.info("🔄 Novo dia detectado, resetando variáveis...")
        time_module.sleep(60)


def agendar_processar_dia_anterior():
    '''agenda o processamento do dia anterior (adicionar no csv)'''
    agora = datetime.now()
    alvo = datetime.combine(agora.date(), datetime.min.time()) + timedelta(hours=0, minutes=5)

    if agora >= alvo:
        alvo += timedelta(days=1)

    delay = (alvo - agora).total_seconds()
    logger.info(f"⏰ Agendando processamento do dia anterior para {alvo}")
    threading.Timer(delay, processar_dia_anterior).start()


def agendar_verificacao_diaria():
    '''agenda a revisão de resultados do dia anterior'''
    agora = datetime.now()
    
    alvo = datetime.combine(agora.date(), time(0, 30))
    
    if agora >= alvo:
        alvo += timedelta(days=1)
    
    delay = (alvo - agora).total_seconds()
    
    logger.info(f"⏰ Agendando verificação diária para {alvo.strftime('%d/%m/%Y %H:%M')}")

    def tarefa():
        logger.info("🔍 Iniciando verificação diária de apostas...")
        try:
            global list_checa, list_uni, resultados_medias
            checa()  # Executa a verificação
            list_checa = []
            list_uni = [0]
        
            try:
                resultados_medias = mlp_pois.calcular_media_gols_por_liga(CSV_FILE, [8, 12])
            except Exception as e:
                logger.error(f"❌ Erro ao calcular médias: {e}")
            
        except Exception as e:
            logger.error(f"❌ Erro na verificação diária: {e}")
        # Reagenda para o próximo dia
        agendar_verificacao_diaria()

    threading.Timer(delay, tarefa).start()

def verificar_aposta(aposta, df_resultados):
    '''função que verifica o resultado de uma aposta'''
    try:
        # Extrair dados da aposta
        id = str(aposta['id'])  # Convert to string to ensure consistent type
        jogo = aposta['jogo']
        linha = aposta.get('linha')
        tipo = aposta.get('tipo')
        time = aposta.get('time')
        mercado = aposta['mercado']
        
        # Log para debug
        print(f"Verificando: ID={id}, mercado={mercado}, time={time}, linha={linha}, tipo={tipo}")
        
        # Verificar se ID existe no DataFrame
        df_resultados['id'] = df_resultados['id'].astype(str)  # Ensure ID column is string
        resultado = df_resultados[df_resultados['id'] == id]
        
        if resultado.empty:
            print(f"ID não encontrado: {id}")
            return None
            
        if mercado == 'goal_line' and not tipo:
            print(f"Aposta ID={id}: tipo não fornecido para goal_line")
            return None

        if (mercado == 'handicap' or mercado == 'draw_no_bet') and not time:
            print(f"Aposta ID={id}: time não fornecido para {mercado}")
            return None
        
        home_time = jogo.split(' X ')[0].strip()
        away_time = jogo.split(' X ')[1].strip()

        row = resultado.iloc[0]

        if mercado == 'over_under':
            if tipo == 'over':
                tipo = 1.0
            else:
                tipo = 2.0
                
            tot_goals = float(row['tot_goals'])
            if pd.isna(tot_goals):
                print(f"Total de gols não disponível para ID={id}")
                return None
                
            if (tipo == 1.0) and (tot_goals > 2.5):
                return 1
            elif (tipo == 1.0) and (tot_goals < 2.5):
                return -1
            elif (tipo == 2.0) and (tot_goals > 2.5):
                return -1
            elif (tipo == 2.0) and (tot_goals < 2.5):
                return 1
            else:
                return 0  # Empate exato em 2.5

        elif mercado == 'goal_line':
            if tipo == 'over':
                tipo = 1.0
            else:
                tipo = 2.0

            # Convert to float and handle NaN values
            type_gl1 = float(row['type_gl1']) if not pd.isna(row['type_gl1']) else None
            type_gl2 = float(row['type_gl2']) if not pd.isna(row['type_gl2']) else None

            if type_gl1 == tipo:
                if row['gl1_positivo']:
                    return 1
                elif row['gl1_negativo']:
                    return -1
                elif row['gl1_reembolso']:
                    return 0
                elif row['gl1_meio_ganho']:
                    return 0.5
                elif row['gl1_meia_perda']:
                    return -0.5
                
            elif type_gl2 == tipo:
                if row['gl2_positivo']:
                    return 1
                elif row['gl2_negativo']:
                    return -1
                elif row['gl2_reembolso']:
                    return 0
                elif row['gl2_meio_ganho']:
                    return 0.5
                elif row['gl2_meia_perda']:
                    return -0.5
                
            print(f"Retornando None para ID={id} porque não encontrou goal_line correspondente")
            return None

        elif mercado == 'handicap':
            if time == home_time:
                time = 1.0
            else:
                time = 2.0

            # Convert to float and handle NaN values
            team_ah1 = float(row['team_ah1']) if not pd.isna(row['team_ah1']) else None
            team_ah2 = float(row['team_ah2']) if not pd.isna(row['team_ah2']) else None

            if team_ah1 == time:
                if row['ah1_positivo']:
                    return 1
                elif row['ah1_negativo']:
                    return -1
                elif row['ah1_reembolso']:
                    return 0
                elif row['ah1_meio_ganho']:
                    return 0.5
                elif row['ah1_meia_perda']:
                    return -0.5
                
            elif team_ah2 == time:
                if row['ah2_positivo']:
                    return 1
                elif row['ah2_negativo']:
                    return -1
                elif row['ah2_reembolso']:
                    return 0
                elif row['ah2_meio_ganho']:
                    return 0.5
                elif row['ah2_meia_perda']:
                    return -0.5
                
            print(f"Retornando None para ID={id} porque não encontrou handicap correspondente")
            return None

        elif mercado == 'draw_no_bet':
            if time == home_time:
                time = 1.0
            else:
                time = 2.0

            # Convert to float and handle NaN values
            draw_no_bet_team1 = float(row['draw_no_bet_team1']) if not pd.isna(row['draw_no_bet_team1']) else None
            draw_no_bet_team2 = float(row['draw_no_bet_team2']) if not pd.isna(row['draw_no_bet_team2']) else None

            if draw_no_bet_team1 == time:
                if row['dnb1_ganha']:
                    return 1
                elif row['dnb1_perde']:
                    return -1
                else:
                    return 0
                
            elif draw_no_bet_team2 == time:
                if row['dnb2_ganha']:
                    return 1
                elif row['dnb2_perde']:
                    return -1
                else:
                    return 0
            
            print(f"Retornando None para ID={id} porque não encontrou draw_no_bet correspondente")
            return None

        elif mercado == 'double_chance':
            if time == home_time:
                time = 1.0
            elif time == away_time:
                time = 2.0
            else:
                time = 3.0

            if time == 1.0:
                if row['res_double_chance1']:
                    return 1
                else:
                    return -1
            elif time == 2.0:
                if row['res_double_chance2']:
                    return 1
                else:
                    return -1
            elif time == 3.0:
                if row['res_double_chance3']:
                    return 1
                else:
                    return -1
            else:
                print(f"Retornando None para ID={id} porque não encontrou double_chance correspondente")
                return None

        print(f"Retornando None para ID={id} porque mercado não reconhecido: {mercado}")
        return None
    except Exception as e:
        print(f"Erro ao verificar aposta: {e}")
        return None

def jogos_do_dia():
    '''função que obtem os jogos do dia'''
    # Obter os dados para o dia anterior e para o dia atual
    ids_anterior, dicio_anterior = apiclient.getAllOlds(leagues=apiclient.leagues_ids, day=dia_anterior())
    ids_atual, dicio_atual = apiclient.getAllOlds(leagues=apiclient.leagues_ids, day=data_hoje)  # Adicionando o dia atual
    
    # Filtrar odds para os jogos do dia anterior e do dia atual
    odds_anterior = apiclient.filtraOddsNovo(ids_anterior)
    odds_atual = apiclient.filtraOddsNovo(ids_atual)
    
    # Transformar os dados de odds
    df_odds_anterior = apiclient.transform_betting_data(odds_anterior)
    df_odds_atual = apiclient.transform_betting_data(odds_atual)
    
    novos_dados = []  # ✅ declarar a lista aqui
    
    # Juntar dados do evento para o dia anterior
    for dados_evento in dicio_anterior:
        event_id = dados_evento.get('id')
        odds_transformadas = df_odds_anterior[df_odds_anterior['id'] == event_id].to_dict('records')
        
        if odds_transformadas:
            merged = {**dados_evento, **odds_transformadas[0], "event_day": dia_anterior()}  # Usando dia anterior
        else:
            merged = {**dados_evento, "event_day": dia_anterior()}  # Usando dia anterior
        
        novos_dados.append(merged)
    
    # Juntar dados do evento para o dia atual
    for dados_evento in dicio_atual:
        event_id = dados_evento.get('id')
        odds_transformadas = df_odds_atual[df_odds_atual['id'] == event_id].to_dict('records')
        
        if odds_transformadas:
            merged = {**dados_evento, **odds_transformadas[0], "event_day": data_hoje}  # Usando dia atual
        else:
            merged = {**dados_evento, "event_day": data_hoje}  # Usando dia atual
        
        novos_dados.append(merged)
    
    # Criando o DataFrame com todos os dados
    df_dados = pd.DataFrame(novos_dados)
    df = df_dados.copy()
    
    # Pré-processamento dos dados
    df = NN.preProcessEstatisticasGerais(df.copy())
    df = NN.preProcessOverUnder(df.copy())
    df = NN.preProcessHandicap_i(df.copy())
    df = NN.preProcessGoalLine_i(df.copy())
    df = NN.preProcessDoubleChance(df.copy())
    df = NN.preProcessDrawNoBet_i(df.copy())
    
    return df


def checa():
    '''função que calcula os resultados das apostas do dia'''
    df_odds = jogos_do_dia()

    resultados_verificados = []
    contador_none = 0
    contador_validos = 0
    # Suponha que você tenha a seguinte lista de dicionários
    
    # Extraindo os valores da chave 'id' em uma nova lista
    # Converte ambos para o mesmo tipo (por exemplo, string)
    df_odds['id'] = df_odds['id'].astype(str)  # Se o id no CSV for string
    ids = [str(dicionario["id"]) for dicionario in list_checa]  # Se a lista de ids for inteira


    df_filtrado = df_odds[df_odds['id'].isin(ids)]


    for aposta in list_checa:
        resultado = verificar_aposta(aposta, df_odds)
        
        # Verifica se o resultado é None/nulo
        if resultado is None:
            contador_none += 1
        else:
            contador_validos += 1
            
        resultados_verificados.append({
            **aposta,
            'resultado': resultado
        })
    
    df_verificacao = pd.DataFrame(resultados_verificados)

    # Conversão para garantir que o 'resultado' seja numérico, com None sendo preservado
    df_verificacao['resultado'] = pd.to_numeric(df_verificacao['resultado'], errors='coerce')

    # Suponha que cada aposta tenha uma coluna 'odd'
    df_verificacao['odd'] = df_verificacao.get('odd')

    # Cálculo das unidades
    df_verificacao['unidade'] = df_verificacao['resultado'].apply(
        lambda x: 1 if x == 1 else
                -1 if x == -1 else
                0.5 if x == 0.5 else
                -0.5 if x == -0.5 else
                0 if x == 0 else
                None
    )


    
    # Cálculo do lucro (None resulta em 0)
    df_verificacao['lucro'] = df_verificacao.apply(
        lambda row: (float(row['odd']) - 1) * row['unidade'] if row['unidade'] == 1 else
                    -1 if row['unidade'] == -1 else
                    (float(row['odd']) - 1) * row['unidade'] if row['unidade'] == 0.5 else
                    -0.5 if row['unidade'] == -0.5 else
                    0, 
        axis=1
    )


 
    total_unidades = df_verificacao['lucro'].sum()
    total_apostas = len(df_verificacao)
    total_apostas_validas = contador_validos
    roi = (total_unidades / total_apostas_validas) * 100 if total_apostas_validas > 0 else 0
    percentual_none = (contador_none / total_apostas) * 100 if total_apostas > 0 else 0

    data_anterior = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    nome_arquivo = f"verificacao_diaria_{data_anterior}.txt"
    global list_uni
  
    resumo_str = (
        f"📊 Estatísticas Detalhadas – {data_anterior}\n"
        f"✅ Total de Apostas: {total_apostas}\n"
        f"✅ Apostas Válidas: {total_apostas_validas}\n"
        f"❓ Apostas None/Nulas: {contador_none} ({percentual_none:.1f}%)\n"
        f"💰 Total de Unidades: {total_unidades:.2f}\n"
        f"📈 ROI (apenas válidas): {roi:.2f}%\n"
    )
    for chat in chats_all:
        tb.sendMessages(chat, resumo_str)
   
    with open(nome_arquivo, "a", encoding="utf-8") as f:
        f.write(f"\n📅 Verificação referente ao dia {data_anterior}\n")
        f.write(f"✅ Total de Apostas: {total_apostas}\n")
        f.write(f"✅ Apostas Válidas: {total_apostas_validas}\n")
        f.write(f"❓ Apostas None/Nulas: {contador_none} ({percentual_none:.1f}%)\n")
        f.write(f"💰 Total de Unidades: {total_unidades:.2f}\n")
        f.write(f"💰 Total de Unidades(acumuladas): {list_uni[0]:.2f}%\n")
        f.write(f"📈 ROI (apenas válidas): {roi:.2f}%\n")
        f.write("-" * 40 + "\n")

    return {
        'dataframe': df_verificacao,
        'total_unidades': total_unidades,
        'roi': roi,
        'apostas_total': total_apostas,
        'apostas_validas': total_apostas_validas,
        'apostas_none': contador_none,
        'percentual_none': percentual_none
    }



'''
def loop_pega_jogos():
   
    while True:
        logger.info("🔎 Buscando jogos programados para hoje...")
        df_jogos = pegaJogosDoDia()
        if not df_jogos.empty:
            logger.info(f"📅 Encontrados {len(df_jogos)} jogos para hoje")
            agenda_processamento(df_jogos)
        else:
            logger.info("ℹ️ Nenhum jogo encontrado por agora")
        time_module.sleep(10 * 60)  
'''
def loop_pega_jogos():
    while True:
        now = datetime.now().time()  # ✅ Corrigido
        start = time(9, 0)
        end = time(21, 0)

        if start <= now <= end:
            logger.info("🔎 Buscando jogos programados para hoje...")
            df_jogos = pegaJogosDoDia()
            if not df_jogos.empty:
                logger.info(f"📅 Encontrados {len(df_jogos)} jogos para hoje")
                agenda_processamento(df_jogos)
            else:
                logger.info("ℹ️ Nenhum jogo encontrado por agora")
            time_module.sleep(10 * 60)  # aguarda 10 minutos
        else:
            logger.info("⏸ Fora do horário de operação. Aguardando para retomar...")
            time_module.sleep(60 * 5)  # verifica a cada 5 minutos



def atualizar_csv_dia_atual():
    COLUNAS_PADRAO = [
        'id', 'event_day', 'home', 'away','league', 'time','home_goals', 'away_goals', 'tot_goals',
        'goals_over_under', 'odd_goals_over1', 'odd_goals_under1',
        'asian_handicap1', 'team_ah1', 'odds_ah1',
        'asian_handicap2', 'team_ah2', 'odds_ah2',
        'goal_line1', 'type_gl1', 'odds_gl1',
        'goal_line2', 'type_gl2', 'odds_gl2',
        'double_chance1', 'odds_dc1',
        'double_chance2', 'odds_dc2',
        'double_chance3', 'odds_dc3',
        'draw_no_bet_team1', 'odds_dnb1',
        'draw_no_bet_team2', 'odds_dnb2',
    ]

    dia = datetime.now().strftime("%Y%m%d")
    logger.info(f"🔄 Atualizando jogos do dia {dia}")

    try:
        # Carregar dados existentes primeiro
        ids_existentes = set()
        if os.path.exists(CSV_FILE):
            df_existente = pd.read_csv(CSV_FILE, dtype={"event_day": str})
            ids_existentes = set(df_existente['id'].astype(str))
            logger.info(f"📊 Total de registros existentes: {len(ids_existentes)}")

        logger.info("🔎 Buscando IDs e dicionário de eventos...")
        ids, dicio = apiclient.getAllOlds(leagues=apiclient.leagues_ids, day=dia)
        logger.info(f"✔️ {len(ids)} eventos encontrados.")

        logger.info("📊 Filtrando e transformando odds...")
        odds_data = apiclient.filtraOddsNovo(ids=ids)
        df_odds = apiclient.transform_betting_data(odds_data)

        novos_dados = []
        for dados_evento in dicio:
            event_id = str(dados_evento.get('id'))
            
            # Verificar se o ID já existe
            if event_id in ids_existentes:
               
                continue
                
            odds_transformadas = df_odds[df_odds['id'] == event_id].to_dict('records')

            if odds_transformadas:
                merged = {**dados_evento, **odds_transformadas[0], "event_day": dia}
            else:
                merged = {**dados_evento, "event_day": dia}

            novos_dados.append(merged)

        if not novos_dados:
            logger.info("⚠️ Nenhum dado novo para adicionar.")
            return

        df_novo = pd.DataFrame(novos_dados)
        logger.info(f"📝 {len(df_novo)} novos registros para adicionar")

        df_novo['id'] = df_novo['id'].astype(str)

        colunas_adicionadas = []
        for coluna in COLUNAS_PADRAO:
            if coluna not in df_novo.columns:
                df_novo[coluna] = None
                colunas_adicionadas.append(coluna)

        if colunas_adicionadas:
            logger.info(f"➕ Colunas adicionadas automaticamente: {', '.join(colunas_adicionadas)}")

        df_novo = df_novo[COLUNAS_PADRAO]

        if os.path.exists(CSV_FILE):
            logger.info("📂 CSV existente encontrado, mesclando dados...")
            df_existente = pd.read_csv(CSV_FILE, dtype={"event_day": str})
            df_existente['id'] = df_existente['id'].astype(str)
            
            duplicatas = set(df_novo['id']).intersection(set(df_existente['id']))
            if duplicatas:
                logger.warning(f"⚠️ Encontrados {len(duplicatas)} IDs que já existem no CSV")
                for dup in duplicatas:
                    logger.warning(f"ID duplicado: {dup}")
            
            df_novo = df_novo[~df_novo['id'].isin(duplicatas)]
            logger.info(f"📝 Após remover duplicatas, {len(df_novo)} registros novos para adicionar")
            
            if len(df_novo) > 0:
                df_final = pd.concat([df_existente, df_novo], ignore_index=True)
                df_final['id'] = df_final['id'].astype(str)
                df_final = df_final.drop_duplicates(subset=['id'], keep='last')
                
                df_final["time"] = df_final["time"].astype(int)
                df_final = df_final.sort_values(by="time", ascending=False).reset_index(drop=True)
                
                duplicatas_finais = df_final[df_final.duplicated(subset=['id'], keep=False)]
                if not duplicatas_finais.empty:
                    logger.warning(f"⚠️ Ainda existem {len(duplicatas_finais)} duplicatas após a concatenação")
                    for id_dup in duplicatas_finais['id'].unique():
                        logger.warning(f"ID duplicado final: {id_dup}")
                
                df_final.to_csv(CSV_FILE, index=False)
                logger.info(f"✅ CSV atualizado com {len(df_novo)} eventos adicionados")
            else:
                logger.info("ℹ️ Nenhum registro novo para adicionar após remoção de duplicatas")
        else:
            logger.info("📄 Nenhum CSV encontrado, criando novo arquivo...")
            df_novo.to_csv(CSV_FILE, index=False)
            logger.info(f"✅ Novo CSV criado com {len(df_novo)} eventos")
        
    except Exception as e:
        logger.error(f"❌ Erro ao atualizar dados do dia {dia}: {type(e).__name__}: {e}")
        raise  

def remover_duplicatas():
    '''remove duplicatas de id no csv'''
    global CSV_FILE
    try:
        logger.info("🔍 Iniciando remoção de duplicatas...")
        df_final = pd.read_csv(CSV_FILE)
        total_antes = len(df_final)
        
        df_final['id'] = df_final['id'].astype(str)
        
        duplicados = df_final[df_final.duplicated(subset=['id'], keep=False)]
        if not duplicados.empty:
            logger.info(f"⚠️ Encontrados {len(duplicados)} registros duplicados:")
            for id_dup in duplicados['id'].unique():
                logger.info(f"ID duplicado: {id_dup}")
        
        df_final = df_final.drop_duplicates(subset=['id'], keep='last')
        
        total_depois = len(df_final)
        removidos = total_antes - total_depois
        
        if removidos > 0:
            logger.info(f"✅ Removidas {removidos} duplicatas")
        
        df_final.to_csv(CSV_FILE, index=False)
        logger.info("✅ CSV atualizado após remoção de duplicatas")
    except Exception as e:
        logger.error(f"❌ Erro ao remover duplicatas: {str(e)}")

def agendar_atualizacao_csv():
    logger.info("🔄 Agendando atualização do CSV...")
    atualizar_csv_dia_atual()
   
    threading.Timer(1800, agendar_atualizacao_csv).start()

def main():
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info("Starting background worker...")
    
 
    threading.Thread(target=checa_virada_do_dia, daemon=True).start()
    

    threading.Thread(target=loop_pega_jogos, daemon=True).start() 
    

 
    agendar_processar_dia_anterior()

    agendar_verificacao_diaria()
    agendar_atualizacao_csv()


    


    while True:
        time_module.sleep(60)  

def pegaJogosDoDia():
    '''pega jogos futuros e programa-os'''
    try:
        dias_para_buscar = [str(data_hoje)]
        if datetime.now().hour >= 20:
            dia_seg = (datetime.now() + timedelta(days=1)).strftime('%Y%m%d')
            dias_para_buscar.append(dia_seg)
        logger.info(f"📅 Dias para buscar: {dias_para_buscar}")

        ids, tempo, nome_time, times_id, league_durations = [], [], [], [], []
        for dia in dias_para_buscar:
            logger.info(f"🔍 Buscando jogos para o dia {dia}")
            r_ids, r_tempo, r_nome_time, r_times_id, r_league_durations = apiclient.getUpcoming(leagues=apiclient.leagues_ids, day=dia)
            logger.info(f"📊 Jogos encontrados: {len(r_ids)}")
            ids.extend(r_ids)
            tempo.extend(r_tempo)
            nome_time.extend(r_nome_time)
            times_id.extend(r_times_id)
            league_durations.extend(r_league_durations)

        if not ids:
            logger.warning("⚠️ Nenhum ID de jogo retornado pela API")
            return pd.DataFrame()

        dados = [{
            "id_jogo": i,
            "horario": h,
            "times": k,
            "home": z,
            "away": t,
            "league_duration": d
        } for i, h, k, (z, t), d in zip(ids, tempo, nome_time, times_id, league_durations)]

        dados_dataframe = pd.DataFrame(dados)
        logger.info(f"📋 Total de jogos antes da filtragem: {len(dados_dataframe)}")
        
        dados_dataframe = dados_dataframe[~dados_dataframe['id_jogo'].isin(programado)]
        logger.info(f"📋 Jogos após remover programados: {len(dados_dataframe)}")

        if dados_dataframe.empty:
            logger.info("ℹ️ Todos os jogos já estão programados")
            return dados_dataframe

        agora = int(time_module.time())
        dados_dataframe['horario'] = dados_dataframe['horario'].astype(int)
        dados_dataframe['send_time'] = dados_dataframe['horario'] - 320
        logger.info(f"⏰ Tempo atual: {agora}")
        logger.info(f"⏰ Primeiro horário de jogo: {dados_dataframe['horario'].min()}")
        logger.info(f"⏰ Primeiro send_time: {dados_dataframe['send_time'].min()}")
        
        dados_dataframe = dados_dataframe[dados_dataframe['send_time'] > (agora - (7 * 60))]
        logger.info(f"📋 Jogos após filtragem por horário: {len(dados_dataframe)}")
        
        dados_dataframe = dados_dataframe.sort_values(by="horario").reset_index(drop=True)
        print(dados_dataframe)

        programados = dados_dataframe['id_jogo'].tolist()
        programado.extend(programados)
        logger.info(f"📌 Adicionados {len(programados)} novos jogos à lista de programados")
        return dados_dataframe

    except Exception as e:
        logger.error(f"❌ Erro ao obter jogos do dia: {str(e)}")
        return pd.DataFrame()




#!roda apos pegajogosDoDia, mas cada acao do jogo sera executada em seu tempo send_timer
def agenda_processamento(df):
    '''programa o processamento do jogo para cerca de 5 minutos antes do evento, e tambem programa a consulta do resultado do jogo'''
    agora = time_module.time()  # timestamp atual em segundos
    logger.info(f"⏳ Agendando {len(df)} eventos...")

    for _, row in df.iterrows():
        delay = row['send_time'] - agora  # tempo até a ação acontecer
        delay = max(0, delay)  # evita delays negativos

        threading.Timer(delay, acao_do_jogo, args=(row,)).start()
        threading.Timer(delay + 1300, checa_jogos_do_dia, args=(row['id_jogo'],)).start()
        print(f"Agendado jogo {row['id_jogo']} para {datetime.fromtimestamp(row['send_time'])}")


def checa_jogos_do_dia(id,tentativa=0):
    '''checa o resultado de um jogo'''
    global list_checa
    print(list_checa)
    df = pd.read_csv(CSV_FILE)
    
    # Garante que 'event_day' é string
    df["event_day"] = df["event_day"].astype(str)

    # Identifica os dois dias mais recentes
    agora = datetime.now()

    # Se a hora for 20h ou mais, pega o dia seguinte e o atual
    if agora.hour >= 20:
        dia1 = agora.strftime("%Y%m%d")
        dia2 = (agora + timedelta(days=1)).strftime("%Y%m%d")
    else:
        dia1 = (agora - timedelta(days=1)).strftime("%Y%m%d")
        dia2 = agora.strftime("%Y%m%d")

    # Filtra o DataFrame
    df_apenas_dois_dias = df[df["event_day"].isin([dia1, dia2])]

    # Pré-processamento dos dados
    df_apenas_dois_dias = NN.preProcessEstatisticasGerais(df_apenas_dois_dias.copy())
    df_apenas_dois_dias = NN.preProcessOverUnder(df_apenas_dois_dias.copy())
    df_apenas_dois_dias = NN.preProcessHandicap_i(df_apenas_dois_dias.copy())
    df_apenas_dois_dias = NN.preProcessGoalLine_i(df_apenas_dois_dias.copy())
    df_apenas_dois_dias = NN.preProcessDoubleChance(df_apenas_dois_dias.copy())
    df_apenas_dois_dias = NN.preProcessDrawNoBet_i(df_apenas_dois_dias.copy())
    

    for a in list_checa:
        if a['id'] == id:
            res = verificar_aposta(a,df_apenas_dois_dias)
            if res is not None:
                if res == 1:
                    a['resultado'] = 'ganhou'
                    a['uni'] = float(a['odd']) - 1
                    list_uni[0] += a['uni']
                    a['unidades acumuladas'] = round(list_uni[0], 2)
                    
                elif res == 0.5:
                    a['resultado'] = 'meio ganho'
                    a['uni'] = float(float(a['odd'] - 1) / 2)
                    list_uni[0] += a['uni']
                    a['unidades acumuladas'] = round(list_uni[0], 2)
                    
                elif res == -0.5:
                    a['resultado'] = 'meia perda'
                    a['uni'] = -0.5
                    list_uni[0] += a['uni']
                    a['unidades acumuladas'] = round(list_uni[0], 2)
                elif res == -1:
                    a['resultado'] = 'perdeu'
                    a['uni'] = -1
                    list_uni[0] += a['uni']
                    a['unidades acumuladas'] = round(list_uni[0], 2)
                else:
                    a['resultado'] = 'empate'
                    a['uni'] = 0
                    a['unidades acumuladas'] = round(list_uni[0], 2)
                a = pd.DataFrame([a])
                a.drop(columns=['id'], inplace=True)
                
                mens = df_para_string(a)
                
                tb.sendMessages(-1002610837223, mens)
            else:
                if tentativa  < 3:
                    time_module.sleep(2000)
                    checa_jogos_do_dia(id,tentativa+1)
                else:
                    logger.info(f"❌ Jogo {id} não retornou resultado após 2 tentativas")
                    return 0

# Função que será executada para cada jogo
def acao_do_jogo(row):
    '''processa o jogo,cria um log e ja envia ao telegram'''
    try:
        global list_checa
        logger.info(f"⚽ Processando jogo {row['id_jogo']}")
        odds = apiclient.filtraOddsNovo([row['id_jogo']])
        if not odds:
            logger.warning(f"⚠️ Nenhuma odd encontrada para o jogo {row['id_jogo']}")
            return 0
        df_odds = apiclient.transform_betting_data(odds)
        
        df_odds['home'] = int(row['home'])
        df_odds['away'] = int(row['away'])
        df_odds['times'] = str(row['times'])
        df_odds['league'] = int(row['league_duration'])
        df_odds['horario'] = row['horario']  # Add game time to df_odds
        
        id = row['id_jogo']
        df_odds = NN.preProcessGeneral_x(df_odds)
    
        lista_bets_a_enviar, listas_para_checar = preve(df_odds, id)
        with open("checar_bets.txt", "a", encoding="utf-8") as f:
            f.write(f"\n--- NOVO LOTE ({datetime.now().isoformat()}) ---\n")
            for aposta in listas_para_checar:
                list_checa.append(aposta)
                
                f.write(json.dumps(aposta, ensure_ascii=False) + "\n")


        if lista_bets_a_enviar:
            logger.info(f"📩 Enviando {len(lista_bets_a_enviar)} previsões para o Telegram")
            for bet in lista_bets_a_enviar:
                for id in chats:
                    tb.sendMessages(id, bet)
        else:
            logger.info("ℹ️ Nenhuma aposta recomendada para este jogo")

    except Exception as e:
        logger.error(f"❌ Erro ao processar jogo {row['id_jogo']}: {str(e)}")
        return 0



def get_first_value(df, col):
    serie = df.get(col)
    if serie is not None and len(serie) > 0 and not pd.isna(serie.iloc[0]):
        return serie.iloc[0]
    return None

def preve(df_linha, id):
    '''faz os preprocessamentos necessarios para cada modelo, chama as funçoes que prevem e escolhe apenas uma a ser enviada'''
    logger.info("🔮 Fazendo previsões para o jogo atual...")
    if df_linha is None or df_linha.empty:
        logger.error("❌ DataFrame de entrada está vazio ou None!")
        return [], []

    try:
        if not lista_th:
            logger.warning("⚠️ Thresholds de modelos ainda não definidos")
            return [], []
        df = mlp_pois.individualiza_jogo_duplo(df_linha, resultados_medias[8], resultados_medias[12])

        df_knn_com_id = mlp_pois.preprocessor(df)
        apostas = mlp_pois.prepara_e_preve(df_knn_com_id, df_linha)
        if apostas is None:
            logger.warning(f"⚠️ Apostas não encontradas para o jogo {id}")
            return [], []
        
        list_true = []
        list_final = []
        list_check = []
        
        # Obter os valores de forma segura usando a nova função
        times = get_first_value(df_linha, 'times')
        horario = get_first_value(df_linha, 'horario')
        
        # Se 'times' ou 'horario' não existirem, não há dados suficientes para continuar
        if times is None or horario is None:
            logger.warning(f"⚠️ Dados essenciais (times ou horario) não encontrados para o jogo {id}")
            return [], []

        home, away = home_e_away(str(times))
        
        if (apostas['ou'] is not None):
            df_ou = pd.DataFrame(columns=['🔔 Jogo','🎯 Liga', '⏰ Horário', '📊 Tipo', '⭐ Odd', '⚽ Linha'])
            df_ou.loc[0, '🔔 Jogo'] = times_para_jogo(str(times))
            df_ou.loc[0, '🎯 Liga'] = f"{get_first_value(df_linha, 'league')} minutos"
            df_ou.loc[0, '📊 Tipo'] = 'over' if apostas['ou'] == 0 else 'under'
            df_ou.loc[0, '⭐ Odd'] = get_first_value(df_linha, 'odd_goals_over1') if df_ou.loc[0, '📊 Tipo'] == 'over' else get_first_value(df_linha, 'odd_goals_under1')
            df_ou.loc[0, '⚽ Linha'] = '2.5'
            df_ou.loc[0, '⏰ Horário'] = datetime.fromtimestamp(horario).strftime('%H:%M')
            list_true.append(df_ou)
            list_check.append({
                'id': id,
                'mercado': 'over_under',
                'tipo': df_ou.loc[0, '📊 Tipo'],
                'linha': '2.5',
                'odd': df_ou.loc[0, '⭐ Odd'],
                'jogo': df_ou.loc[0, '🔔 Jogo']
            })

        if (apostas['h'] is not None):
            df_h = pd.DataFrame(columns=['🔔 Jogo','🎯 Liga','⏰ Horário','🚀 time', '⭐ Odd', '⚽ handicap'])
            df_h.loc[0, '🔔 Jogo'] = times_para_jogo(str(times))
            df_h.loc[0, '🎯 Liga'] = f"{get_first_value(df_linha, 'league')} minutos"
            if apostas['h'] == 0:
                ah1 = str(get_first_value(df_linha, 'asian_handicap1_1'))
                ah2 = str(get_first_value(df_linha, 'asian_handicap1_2'))
            else:
                ah1 = str(get_first_value(df_linha, 'asian_handicap2_1'))
                ah2 = str(get_first_value(df_linha, 'asian_handicap2_2'))
            df_h.loc[0, '⚽ handicap'] = ah1 if ah1 == ah2 else f"{ah1} , {ah2}"
            df_h.loc[0, '🚀 time'] = home if apostas['h'] == 0 else away
            odd_ah1 = get_first_value(df_linha, 'odds_ah1')
            odd_ah2 = get_first_value(df_linha, 'odds_ah2')
            df_h.loc[0, '⭐ Odd'] = str(odd_ah1 if df_h.loc[0, '🚀 time'] == home else odd_ah2)
            df_h.loc[0, '⏰ Horário'] = datetime.fromtimestamp(horario).strftime('%H:%M')
            list_true.append(df_h)
            list_check.append({
                'id': id,
                'mercado': 'handicap',
                'time': df_h.loc[0, '🚀 time'],
                'linha': df_h.loc[0, '⚽ handicap'],
                'odd': df_h.loc[0, '⭐ Odd'],
                'jogo': df_h.loc[0, '🔔 Jogo']
            })
        if (apostas['gl'] is not None):
            df_gl = pd.DataFrame(columns=['🔔 Jogo','🎯 Liga', '⏰ Horário', '📊 Tipo', '⭐ Odd', '⚽ Linha'])
            df_gl.loc[0, '🔔 Jogo'] = times_para_jogo(str(times))
            df_gl.loc[0, '🎯 Liga'] = f"{get_first_value(df_linha, 'league')} minutos"
            gl1 = str(get_first_value(df_linha, 'goal_line1_1'))
            gl2 = str(get_first_value(df_linha, 'goal_line1_2'))
            df_gl.loc[0, '⚽ Linha'] = gl1 if gl1 == gl2 else f"{gl1} , {gl2}"
            df_gl.loc[0, '📊 Tipo'] = 'over' if apostas['gl'] == 0 else 'under'
            odd_gl1 = get_first_value(df_linha, 'odds_gl1')
            odd_gl2 = get_first_value(df_linha, 'odds_gl2')
            df_gl.loc[0, '⭐ Odd'] = str(odd_gl1 if df_gl.loc[0, '📊 Tipo'] == 'over' else odd_gl2)
            df_gl.loc[0, '⏰ Horário'] = datetime.fromtimestamp(horario).strftime('%H:%M')
            list_true.append(df_gl)
            list_check.append({
                'id': id,
                'mercado': 'goal_line',
                'tipo': df_gl.loc[0, '📊 Tipo'],
                'linha': df_gl.loc[0, '⚽ Linha'],
                'odd': df_gl.loc[0, '⭐ Odd'],
                'jogo': df_gl.loc[0, '🔔 Jogo']
            })

        if (apostas['dc'] is not None):
            df_dc = pd.DataFrame(columns=['🔔 Jogo','🎯 Liga','⏰ Horário', '📊 Double Chance', '⭐ Odd'])
            df_dc.loc[0, '🔔 Jogo'] = times_para_jogo(str(times))
            df_dc.loc[0, '🎯 Liga'] = f"{get_first_value(df_linha, 'league')} minutos"

            if apostas['dc'] == 0:
                df_dc.loc[0, '📊 Double Chance'] = home
                df_dc.loc[0, '⭐ Odd'] = get_first_value(df_linha, 'odds_dc1')
            elif apostas['dc'] == 1:
                df_dc.loc[0, '📊 Double Chance'] = away
                df_dc.loc[0, '⭐ Odd'] = get_first_value(df_linha, 'odds_dc2')
            else:
                df_dc.loc[0, '📊 Double Chance'] = f"{home} ou {away}"
                df_dc.loc[0, '⭐ Odd'] = get_first_value(df_linha, 'odds_dc3')

            df_dc.loc[0, '⏰ Horário'] = datetime.fromtimestamp(horario).strftime('%H:%M')

            list_true.append(df_dc)
            list_check.append({
                'id': id,
                'mercado': 'double_chance',
                'time': df_dc.loc[0, '📊 Double Chance'],
                'odd': df_dc.loc[0, '⭐ Odd'],
                'jogo': df_dc.loc[0, '🔔 Jogo']
            })
        if (apostas['dnb'] is not None):
            df_dnb = pd.DataFrame(columns=['🔔 Jogo','🎯 Liga', '⏰ Horário', '📊 Draw No Bet', '⭐ Odd'])
            df_dnb.loc[0, '🔔 Jogo'] = times_para_jogo(str(times))
            df_dnb.loc[0, '🎯 Liga'] = f"{get_first_value(df_linha, 'league')} minutos"
            df_dnb.loc[0, '📊 Draw No Bet'] = home if apostas['dnb'] == 0 else away

            odd_dnb1 = get_first_value(df_linha, 'odds_dnb1')
            odd_dnb2 = get_first_value(df_linha, 'odds_dnb2')
            df_dnb.loc[0, '⭐ Odd'] = odd_dnb1 if apostas['dnb'] == 0 else odd_dnb2
            df_dnb.loc[0, '⏰ Horário'] = datetime.fromtimestamp(horario).strftime('%H:%M')

            list_true.append(df_dnb)
            list_check.append({
                'id': id,
                'mercado': 'draw_no_bet',
                'time': df_dnb.loc[0, '📊 Draw No Bet'],
                'odd': df_dnb.loc[0, '⭐ Odd'],
                'jogo': df_dnb.loc[0, '🔔 Jogo']
            })

        for df in list_true:
            df = pd.DataFrame(df)
            men = df_para_string(df)
            list_final.append(men)

        if list_final:
            logger.info("✅ Previsões recomendadas:")
            for recomendacao in list_final:
                logger.info(f"👉 {recomendacao}")
        else:
            logger.info("❌ Nenhuma previsão foi considerada válida.")

        return list_final, list_check

    except Exception as e:
        logger.error(f"❌ Erro durante a previsão: {str(e)}")
        return [], []

def times_para_jogo(times):
    '''pega os times e estiliza-os para formato de confronto (A X B)'''
    #('arsenal','mai')
    c = times.find(',')
    time_a = times[2:c-1]
    time_b = times[c+3:-2]
    final = f'{time_a.upper()} X {time_b.upper()}'
    return final

def home_e_away(times):
    '''limpa os nomes dos times'''
    #('arsenal','mai')
    c = times.find(',')
    time_a = times[2:c-1].upper()
    time_b = times[c+3:-2].upper()
    return time_a, time_b

def df_para_string(df):
    '''pega um df e transforma em string para ser enviado ao telegram'''
    mensagens = []

    for _, row in df.iterrows():
        msg = ""
   
        cont=0
        for col in df.columns:
            if cont == 0:
                msg += f"{col}: {row[col]}\n\n"
            else:
                msg += f"{col}: {row[col]}\n"
            cont+=1
        mensagens.append(msg.strip())

    return mensagens



def processar_dia_anterior():
    '''adiciona todos os jogos do dia passado ao csv'''
    COLUNAS_PADRAO = [
        'id', 'event_day', 'home', 'away','league','time', 'home_goals', 'away_goals', 'tot_goals',
        'goals_over_under', 'odd_goals_over1', 'odd_goals_under1',
        'asian_handicap1', 'team_ah1', 'odds_ah1',
        'asian_handicap2', 'team_ah2', 'odds_ah2',
        'goal_line1', 'type_gl1', 'odds_gl1',
        'goal_line2', 'type_gl2', 'odds_gl2',
        'double_chance1', 'odds_dc1',
        'double_chance2', 'odds_dc2',
        'double_chance3', 'odds_dc3',
        'draw_no_bet_team1', 'odds_dnb1',
        'draw_no_bet_team2', 'odds_dnb2',
    ]
    dia = dia_anterior()
    print(f"🔄 Processando jogos do dia {dia}")

    try:
        print("🔎 Buscando IDs e dicionário de eventos...")
        ids, dicio = apiclient.getAllOlds(leagues=apiclient.leagues_ids, day=dia)
        print(f"✔️ {len(ids)} eventos encontrados.")

        print("📊 Filtrando e transformando odds...")
        odds_data = apiclient.filtraOddsNovo(ids=ids)
        df_odds = apiclient.transform_betting_data(odds_data)  

        novos_dados = []
        for dados_evento in dicio:
            event_id = dados_evento.get('id')
            odds_transformadas = df_odds[df_odds['id'] == event_id].to_dict('records')

            if odds_transformadas:
                merged = {**dados_evento, **odds_transformadas[0], "event_day": dia}
            else:
                merged = {**dados_evento, "event_day": dia}

            novos_dados.append(merged)

        if novos_dados:
            print(f"🧩 {len(novos_dados)} eventos com odds processados.")
            df_novo = pd.DataFrame(novos_dados)

            colunas_adicionadas = []
            for coluna in COLUNAS_PADRAO:
                if coluna not in df_novo.columns:
                    df_novo[coluna] = None
                    colunas_adicionadas.append(coluna)

            if colunas_adicionadas:
                print(f"➕ Colunas adicionadas automaticamente: {', '.join(colunas_adicionadas)}")

            df_novo = df_novo[COLUNAS_PADRAO]

            if os.path.exists(CSV_FILE):
                print("📂 CSV existente encontrado, mesclando dados...")
                df_existente = pd.read_csv(CSV_FILE, dtype={"event_day": str})
                df_final = pd.concat([df_existente, df_novo], ignore_index=True)
                df_final["time"] = df_final["time"].astype(int)
                df_final = df_final.sort_values(by="time", ascending=False).reset_index(drop=True)

                primeiro_dia = df_final["event_day"].min()
                #df_final = df_final[df_final["event_day"] != primeiro_dia]
            else:
                print("📄 Nenhum CSV encontrado, criando novo arquivo...")
                df_final = df_novo
                primeiro_dia = "N/A"

            df_final.to_csv(CSV_FILE, index=False)
            remover_duplicatas()
            print(f"✅ Dados atualizados com sucesso! Dia {primeiro_dia} removido, dia {dia} adicionado.")
        else:
            print(f"⚠️ Nenhum dado encontrado para o dia {dia}")

    except Exception as e:
        print(f"❌ Erro ao processar dia {dia}: {type(e).__name__}: {e}")




# Variável global para controlar o estado do servidor
server_running = True

def signal_handler(signum, frame):
    logger.info("Received shutdown signal. Cleaning up...")
    sys.exit(0)

if __name__ == "__main__":
    main()

import pandas as pd
import os
from datetime import datetime

def atualizar_csv_dia_atual():
    COLUNAS_PADRAO = [
        'id', 'event_day', 'home', 'away','league','time', 'home_goals', 'away_goals', 'tot_goals',
        'goals_over_under', 'odd_goals_over1', 'odd_goals_under1',
        'asian_handicap1', 'team_ah1', 'odds_ah1',
        'asian_handicap2', 'team_ah2', 'odds_ah2',
        'goal_line1', 'type_gl1', 'odds_gl1',
        'goal_line2', 'type_gl2', 'odds_gl2',
        'double_chance1', 'odds_dc1',
        'double_chance2', 'odds_dc2',
        'double_chance3', 'odds_dc3',
        'draw_no_bet_team1', 'odds_dnb1',
        'draw_no_bet_team2', 'odds_dnb2',
    ]

    dias_para_buscar = [datetime.now().strftime("%Y%m%d")]
    if datetime.now().hour >= 20:
        dia_seguinte = (datetime.now() + timedelta(days=1)).strftime('%Y%m%d')
        dias_para_buscar.append(dia_seguinte)
    
    logger.info(f"📅 Dias para buscar: {dias_para_buscar}")

    try:
      
        ids_existentes = set()
        if os.path.exists(CSV_FILE):
            df_existente = pd.read_csv(CSV_FILE, dtype={"event_day": str})
            ids_existentes = set(df_existente['id'].astype(str))
            logger.info(f"📊 Total de registros existentes: {len(ids_existentes)}")

        todos_ids = []
        todos_dicionarios = []
        
        for dia in dias_para_buscar:
            logger.info(f"🔎 Buscando IDs e dicionário de eventos para o dia {dia}...")
            ids, dicio = apiclient.getAllOlds(leagues=apiclient.leagues_ids, day=dia)
            logger.info(f"✔️ {len(ids)} eventos encontrados para o dia {dia}.")
            
            for evento in dicio:
                evento['event_day'] = dia
            
            todos_ids.extend(ids)
            todos_dicionarios.extend(dicio)

        logger.info(f"✔️ Total de {len(todos_ids)} eventos encontrados em todos os dias.")

        if not todos_ids:
            logger.info("⚠️ Nenhum evento encontrado para os dias especificados.")
            return

        logger.info("📊 Filtrando e transformando odds...")
        odds_data = apiclient.filtraOddsNovo(ids=todos_ids)
        df_odds = apiclient.transform_betting_data(odds_data)

        novos_dados = []
        for dados_evento in todos_dicionarios:
            event_id = str(dados_evento.get('id'))
            
            if event_id in ids_existentes:
                logger.info(f"⚠️ ID {event_id} já existe no CSV, pulando...")
                continue
                
            odds_transformadas = df_odds[df_odds['id'] == event_id].to_dict('records')

            if odds_transformadas:
                merged = {**dados_evento, **odds_transformadas[0]}
            else:
                merged = dados_evento.copy()

            novos_dados.append(merged)

        if not novos_dados:
            logger.info("⚠️ Nenhum dado novo para adicionar.")
            return

        df_novo = pd.DataFrame(novos_dados)
        logger.info(f"📝 {len(df_novo)} novos registros para adicionar")

        df_novo['id'] = df_novo['id'].astype(str)

        colunas_adicionadas = []
        for coluna in COLUNAS_PADRAO:
            if coluna not in df_novo.columns:
                df_novo[coluna] = None
                colunas_adicionadas.append(coluna)

        if colunas_adicionadas:
            logger.info(f"➕ Colunas adicionadas automaticamente: {', '.join(colunas_adicionadas)}")

        df_novo = df_novo[COLUNAS_PADRAO]

        if os.path.exists(CSV_FILE):
            logger.info("📂 CSV existente encontrado, mesclando dados...")
            df_existente = pd.read_csv(CSV_FILE, dtype={"event_day": str})
            df_existente['id'] = df_existente['id'].astype(str)
            
            duplicatas = set(df_novo['id']).intersection(set(df_existente['id']))
            if duplicatas:
                logger.warning(f"⚠️ Encontrados {len(duplicatas)} IDs que já existem no CSV")
                for dup in duplicatas:
                    logger.warning(f"ID duplicado: {dup}")
            
            df_novo = df_novo[~df_novo['id'].isin(duplicatas)]
            logger.info(f"📝 Após remover duplicatas, {len(df_novo)} registros novos para adicionar")
            
            if len(df_novo) > 0:
                df_final = pd.concat([df_existente, df_novo], ignore_index=True)
                df_final['id'] = df_final['id'].astype(str)
                df_final = df_final.drop_duplicates(subset=['id'], keep='last')
                
                df_final["time"] = df_final["time"].astype(int)
                df_final = df_final.sort_values(by="time", ascending=False).reset_index(drop=True)
                
                duplicatas_finais = df_final[df_final.duplicated(subset=['id'], keep=False)]
                if not duplicatas_finais.empty:
                    logger.warning(f"⚠️ Ainda existem {len(duplicatas_finais)} duplicatas após a concatenação")
                    for id_dup in duplicatas_finais['id'].unique():
                        logger.warning(f"ID duplicado final: {id_dup}")
                
                df_final.to_csv(CSV_FILE, index=False)
                logger.info(f"✅ CSV atualizado com {len(df_novo)} eventos adicionados")
            else:
                logger.info("ℹ️ Nenhum registro novo para adicionar após remoção de duplicatas")
        else:
            logger.info("📄 Nenhum CSV encontrado, criando novo arquivo...")
            df_novo.to_csv(CSV_FILE, index=False)
            logger.info(f"✅ Novo CSV criado com {len(df_novo)} eventos")
        
    except Exception as e:
        logger.error(f"❌ Erro ao atualizar dados dos dias {dias_para_buscar}: {type(e).__name__}: {e}")
        raise 
    

