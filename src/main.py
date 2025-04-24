import time as time_module  # para ainda usar time.sleep
from datetime import datetime, timedelta, time

import tensorflow as tf
from api import BetsAPIClient, dia_anterior
import pandas as pd
from dotenv import load_dotenv
import os
import threading
import NN
import telegramBot as tb
import logging
import json

#tentar arrumar as estatisticas
#ver os team_ah e type_gl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

#NORMALIZAR OS DADOS


load_dotenv()

api = os.getenv("API_KEY")
chat_id = int(os.getenv("CHAT_ID"))
# -1002610837223
chats = [chat_id, -1002610837223]



apiclient = BetsAPIClient(api_key=api)

#talvez adicionar liga como parametro da NN (one hot ou label encoder?), talvez nao normalizar a linha do handicap?, definir um th bem menor que o esperado? melhorar os retornos?

#df = pd.read_csv('src\resultados_novo.csv')
#CSV_FILE = r"C:\Users\Leoso\Downloads\projBotAposta\src\resultados_novo.csv"
CSV_FILE = 'resultados_60_ofc.csv'
#lista dos thresholds das nns
lista_th = [0.575,0.4,0.5,0.5,0.575,0.5]
list_checa = []



#!pega uma vez ao virar o dia e depois de 20 em 20 min pra testar
data_hoje = datetime.now().date().strftime('%Y%m%d')
#!reseta todo dia as 00:00, guarda jogos programados do dia, deve zerar ao mudar o dia
programado = []

def checa_virada_do_dia():
    global data_hoje, programado
    while True:
        novo_dia = datetime.now().date().strftime('%Y%m%d')
        if novo_dia != data_hoje:
            data_hoje = novo_dia
            programado = []
            logger.info("🔄 Novo dia detectado, resetando variáveis...")
        time_module.sleep(60)


def agendar_processar_dia_anterior():
    agora = datetime.now()
    alvo = datetime.combine(agora.date(), datetime.min.time()) + timedelta(hours=0, minutes=5)

    if agora >= alvo:
        alvo += timedelta(days=1)

    delay = (alvo - agora).total_seconds()
    logger.info(f"⏰ Agendando processamento do dia anterior para {alvo}")
    threading.Timer(delay, processar_dia_anterior).start()


def agendar_criacao_nns():
    agora = datetime.now()
    alvo = datetime.combine(agora.date(), datetime.min.time()) + timedelta(hours=0, minutes=15)

    # Se o horário atual já passou de 00:15, agendar para o próximo dia
    if agora >= alvo:
        alvo += timedelta(days=1)

    delay = (alvo - agora).total_seconds()
    logger.info(f"⏰ Agendando criação de NNs para {alvo}")

    def tarefa():
        logger.info("🧠 Iniciando criação de modelos de rede neural...")
        
        criaTodasNNs()
        logger.info("✅ Modelos de rede neural criados com sucesso")
        print(lista_th)

    threading.Timer(delay, tarefa).start()

def agendar_verificacao_diaria():
    agora = datetime.now()
    
    # Define o horário alvo (00:30)
    alvo = datetime.combine(agora.date(), time(0, 30))
    
    # Se já passou das 00:30 hoje, agenda para amanhã
    if agora >= alvo:
        alvo += timedelta(days=1)
    
    # Calcula o delay em segundos
    delay = (alvo - agora).total_seconds()
    
    logger.info(f"⏰ Agendando verificação diária para {alvo.strftime('%d/%m/%Y %H:%M')}")

    def tarefa():
        logger.info("🔍 Iniciando verificação diária de apostas...")
        try:
            global list_checa
            checa()  # Executa a verificação
            list_checa = []

        except Exception as e:
            logger.error(f"❌ Erro na verificação diária: {e}")
        
        # Reagenda para o próximo dia
        agendar_verificacao_diaria()

    # Cria o timer
    threading.Timer(delay, tarefa).start()


def verificar_aposta(aposta, df_resultados):
    try:
        id = aposta['id']
        jogo = aposta['jogo']
        linha = aposta.get('linha')
        tipo = aposta.get('tipo')
        time = aposta.get('time')
        mercado = aposta['mercado']

        home_time, away_time = [t.strip() for t in jogo.split(' X ')]

        resultado = df_resultados[
            (df_resultados['id'] == id)
        ]

        if resultado.empty:
            return None

        row = resultado.iloc[0]

        if mercado == 'goal_line':
            # Comparar float e tipo (over/under)
            if tipo == 'over':
                tipo = 1.0
            else:
                tipo = 2.0
                
            if row['goal_line1'] == str(linha) and row['type_gl1'] == tipo:
                return row['gl1_positivo']
            elif row['goal_line2'] == str(linha) and row['type_gl2'] == tipo:
                return row['gl2_positivo']
            else:
                return None

        elif mercado == 'handicap':
            if time == home_time:
                time = 1.0
            else:
                time = 2.0
            def format_handicap(valor):
                partes = str(valor).replace(' ', '').split(',')
                if len(partes) == 2 and partes[0] == partes[1]:
                    return partes[0]  # Ex: "0.0, 0.0" → "0.0"
                return str(valor).strip()
            if row['team_ah1'] == time and format_handicap(row['asian_handicap1']) == linha:
                return row['ah1_positivo']
            elif row['team_ah2'] == time and format_handicap(row['asian_handicap2']) == linha:
                return row['ah2_positivo']
            else:
                return None

        elif mercado == 'draw_no_bet':
            if time == home_time:
                time = 1.0
            else:
                time = 2.0
            if row['draw_no_bet_team1'] == time:
                return row['dnb1_ganha']
            elif row['draw_no_bet_team2'] == time:
                return row['dnb2_ganha']
            else:
                return None
        elif mercado == 'double_chance':
            if time == home_time:
                time = 1.0
            elif time == away_time:
                time = 2.0
            else:
                time = 3.0
            if time == 1.0:
                return row['res_double_chance1']
            elif time == 2.0:
                return row['res_double_chance2']
            elif time == 3.0:
                return row['res_double_chance3']

            else:
                return None

        return None
    except Exception as e:
        print(f"Erro ao verificar aposta: {e}")
        return None
def jogos_do_dia():
    ids, dicio = apiclient.getAllOlds(leagues=apiclient.leagues_ids, day=dia_anterior())
    #pegar dia atual tambem
    odds = apiclient.filtraOddsNovo(ids)
    df_odds = apiclient.transform_betting_data(odds)
    novos_dados = []  # ✅ declarar a lista aqui
    
    # Juntar com dados do evento
    for dados_evento in dicio:
        event_id = dados_evento.get('id')
        odds_transformadas = df_odds[df_odds['id'] == event_id].to_dict('records')
        
        if odds_transformadas:
            merged = {**dados_evento, **odds_transformadas[0], "event_day": data_hoje}
        else:
            merged = {**dados_evento, "event_day": data_hoje}
        
        novos_dados.append(merged)
    
    df_dados = pd.DataFrame(novos_dados)
    df = df_dados.copy()
    df = NN.preProcessEstatisticasGerais(df.copy())
    df = NN.preProcessOverUnder(df.copy())
    df = NN.preProcessHandicap(df.copy())
    df = NN.preProcessGoalLine(df.copy())
    df = NN.preProcessDoubleChance(df.copy())
    df = NN.preProcessDrawNoBet(df.copy())
    return df


def checa():
    df_odds = jogos_do_dia()
    resultados_verificados = []
    contador_none = 0
    contador_validos = 0

    for aposta in list_checa:
        resultado = verificar_aposta(aposta, df_odds)
        
        # Verifica se o resultado é None/nulo
        if resultado is None or pd.isna(resultado):
            contador_none += 1
        else:
            contador_validos += 1
            
        resultados_verificados.append({
            **aposta,
            'resultado': resultado
        })
    

    df_verificacao = pd.DataFrame(resultados_verificados)

    # Conversão para facilitar o cálculo (mantém None para resultados inválidos)
    df_verificacao['resultado'] = pd.to_numeric(df_verificacao['resultado'], errors='coerce')

    # Suponha que cada aposta tenha uma coluna 'odd'
    df_verificacao['odd'] = df_verificacao.get('odd')

    # Cálculo das unidades (considera explicitamente None)
    df_verificacao['unidade'] = df_verificacao['resultado'].apply(
        lambda x: 1 if x in [1, 1.0, True] else (0 if x in [0, 0.0, False] else None)
    )
    
    # Cálculo do lucro (None resulta em 0)
    df_verificacao['lucro'] = df_verificacao.apply(
        lambda row: (row['odd'] - 1) if row['unidade'] == 1 
                   else (-1 if row['unidade'] == 0 
                   else 0), 
        axis=1
    )

    # Estatísticas
    total_unidades = df_verificacao['lucro'].sum()
    total_apostas = len(df_verificacao)
    total_apostas_validas = contador_validos
    roi = (total_unidades / total_apostas_validas) * 100 if total_apostas_validas > 0 else 0
    percentual_none = (contador_none / total_apostas) * 100 if total_apostas > 0 else 0
    # Pega o dia anterior
    data_anterior = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    nome_arquivo = f"verificacao_diaria_{data_anterior}.txt"

        # Gera string de resumo para envio ao bot
    resumo_str = (
        f"📊 Estatísticas Detalhadas – {data_anterior}\n"
        f"✅ Total de Apostas: {total_apostas}\n"
        f"✅ Apostas Válidas: {total_apostas_validas}\n"
        f"❓ Apostas None/Nulas: {contador_none} ({percentual_none:.1f}%)\n"
        f"💰 Total de Unidades: {total_unidades:.2f}\n"
        f"📈 ROI (apenas válidas): {roi:.2f}%\n"
    )
    for chat in chats:
        tb.sendMessages(chat, resumo_str)
   

    with open(nome_arquivo, "a", encoding="utf-8") as f:
        f.write(f"\n📅 Verificação referente ao dia {data_anterior}\n")
        f.write(f"✅ Total de Apostas: {total_apostas}\n")
        f.write(f"✅ Apostas Válidas: {total_apostas_validas}\n")
        f.write(f"❓ Apostas None/Nulas: {contador_none} ({percentual_none:.1f}%)\n")
        f.write(f"💰 Total de Unidades: {total_unidades:.2f}\n")
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
resultados_verificados = []

for aposta in lista_checa:
    resultado = verificar_aposta(aposta, df_resultados)
    resultados_verificados.append({
        **aposta,
        'resultado': resultado
    })

df_verificacao = pd.DataFrame(resultados_verificados)
print(df_verificacao[['jogo', 'mercado', 'linha', 'tipo', 'time', 'resultado']])

'''

def loop_pega_jogos():
    while True:
        logger.info("🔎 Buscando jogos programados para hoje...")
        df_jogos = pegaJogosDoDia()
        if not df_jogos.empty:
            logger.info(f"📅 Encontrados {len(df_jogos)} jogos para hoje")
            pegaOddsEvento(df_jogos)
        else:
            logger.info("ℹ️ Nenhum jogo encontrado por agora")
        time_module.sleep(10 * 60)  # 20 minutos

def main():
    logger.info("🚀 Iniciando aplicação de apostas esportivas")
    # Thread para verificar virada do dia e resetar programado
    threading.Thread(target=checa_virada_do_dia, daemon=True).start()

    agendar_verificacao_diaria()


    # Agendar processamento do dia anterior
    agendar_processar_dia_anterior()

    # Agendar criação das NNs
    agendar_criacao_nns()

    # Iniciar loop que pega os jogos de tempos em tempos
    threading.Thread(target=loop_pega_jogos, daemon=True).start()
    tb.start_bot()
    logger.info("✅ Todos os serviços foram iniciados")
    
def pegaJogosDoDia():
    try:
        dias_para_buscar = [str(data_hoje)]
        if datetime.now().hour >= 20:
            # Se já for 21h (ou depois), pega também os jogos de amanhã
            dia_seg = (datetime.now() + timedelta(days=1)).strftime('%Y%m%d')
            dias_para_buscar.append(dia_seg)

        ids, tempo, nome_time, times_id = [], [], [], [], 

        for dia in dias_para_buscar:
            r_ids, r_tempo, r_nome_time, r_times_id = apiclient.getUpcoming(leagues=apiclient.leagues_ids, day=dia)
            ids.extend(r_ids)
            tempo.extend(r_tempo)
            nome_time.extend(r_nome_time)
            times_id.extend(r_times_id)

        if not ids:
            logger.warning("⚠️ Nenhum ID de jogo retornado pela API")
            return pd.DataFrame()

        dados = [{
            "id_jogo": i,
            "horario": h,
            "times": k,
            "home": z,
            "away": t,
   
        } for i, h, k, (z, t) in zip(ids, tempo, nome_time, times_id)]
        print(dados)
        dados_dataframe = pd.DataFrame(dados)
        dados_dataframe = dados_dataframe[~dados_dataframe['id_jogo'].isin(programado)]


        if dados_dataframe.empty:
            logger.info("ℹ️ Todos os jogos já estão programados")
            return dados_dataframe

        agora = int(time_module.time())

        dados_dataframe['horario'] = dados_dataframe['horario'].astype(int)
        dados_dataframe['send_time'] = dados_dataframe['horario'] - 320
        dados_dataframe = dados_dataframe[dados_dataframe['send_time'] > (agora - (7 * 60))]
        dados_dataframe = dados_dataframe.sort_values(by="horario").reset_index(drop=True)

        programados = dados_dataframe['id_jogo'].tolist()
        programado.extend(programados)
        logger.info(f"📌 Adicionados {len(programados)} novos jogos à lista de programados")
        return dados_dataframe

    except Exception as e:
        logger.error(f"❌ Erro ao obter jogos do dia: {str(e)}")
        return pd.DataFrame()




#!roda apos pegajogosDoDia, mas cada acao do jogo sera executada em seu tempo send_timer
def pegaOddsEvento(df):
    agora = time_module.time()  # timestamp atual em segundos
    logger.info(f"⏳ Agendando {len(df)} eventos...")

    for _, row in df.iterrows():
        delay = row['send_time'] - agora  # tempo até a ação acontecer
        delay = max(0, delay)  # evita delays negativos

        threading.Timer(delay, acao_do_jogo, args=(row,)).start()
        print(f"Agendado jogo {row['id_jogo']} para {datetime.fromtimestamp(row['send_time'])}")




# Função que será executada para cada jogo
def acao_do_jogo(row):
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
        
        id = row['id_jogo']
        df_odds = NN.preProcessGeneral_x(df_odds)
        lista_bets_a_enviar, listas_para_checar = preve(df_odds, id)
        with open("checar_bets.txt", "a", encoding="utf-8") as f:
            f.write(f"\n--- NOVO LOTE ({datetime.now().isoformat()}) ---\n")
            for aposta in listas_para_checar:
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

#! roda todo dia as 00:15
def criaTodasNNs():
    global lista_th 
    logger.info("🔧 Criando todos os modelos de rede neural...")
    lista_th = NN.criaNNs()
    logger.info(f"📊 Thresholds definidos: {lista_th}")

def preve(df_linha, id):

    logger.info("🔮 Fazendo previsões para o jogo atual...")
    try:
        if not lista_th:
            logger.warning("⚠️ Thresholds de modelos ainda não definidos")
            return []
        prepOverUnder, dados_OU = NN.prepNNOver_under_X(df_linha)

        if prepOverUnder is None:
            tipo_over_under, res_under_over = None, None
        else:
            tipo_over_under, res_under_over = predicta_over_under(prepOverUnder, dados_OU)
        
        prepHandicap, dados_ah = NN.prepNNHandicap_X(df_linha)

        if prepHandicap is None:
            time_handicap, res_handicap = None, None
        else:
            time_handicap, res_handicap = predicta_handicap(prepHandicap, dados_ah)
        
        prepGoal_line, dados_gl = NN.prepNNGoal_line_X(df_linha)

        if prepGoal_line is None:
            linha_gl, res_goal_line = None, None
        else:
            linha_gl, res_goal_line = predicta_goal_line(prepGoal_line, dados_gl)
        
        prepDouble_chance, dados_dc = NN.prepNNDouble_chance_X(df_linha)

        if prepDouble_chance is None:
            type_dc, res_double_chance = None, None
        else:
            type_dc, res_double_chance = predicta_double_chance(prepDouble_chance, dados_dc)
        
        prepDraw_no_bet, dados_dnb = NN.prepNNDraw_no_bet_X(df_linha)

        if prepDraw_no_bet is None:
            time_draw_no_bet, res_draw_no_bet = None, None
        else:
            time_draw_no_bet, res_draw_no_bet = predicta_draw_no_bet(prepDraw_no_bet, dados_dnb)

        lista_preds_true = [tipo_over_under, res_under_over, time_handicap, res_handicap, linha_gl, res_goal_line, type_dc, res_double_chance,time_draw_no_bet, res_draw_no_bet]
        
        logger.info(f"🧠 Predições retornadas: {lista_preds_true}")

        list_res = []
        
        if res_under_over:
            list_res.append(res_under_over)
        if res_handicap:
            list_res.append(res_handicap)
        if res_goal_line:
            list_res.append(res_goal_line)
        if res_double_chance:
            list_res.append(res_double_chance)
        if res_draw_no_bet:
            list_res.append(res_draw_no_bet)
        print(list_res)
        melhor = None

        try:
            melhor = max(list_res)
            print(melhor)
        except:
            logger.info("❌ Nenhuma previsão foi considerada válida.")

        #[(0) - over, (1) - under, (2) - handicap home, (3) - handicap away, (4) - over goal_line, (5) - under goal_line, (6) dc1, dc2, dc3, dnb1, dnb2]
        list_true = []
        list_final = []
        list_check = []
        if melhor:
            if (tipo_over_under is not None and (res_under_over == melhor)):
                dados_OU['🔔 Jogo'] = times_para_jogo(str(dados_OU['times'].iloc[0]))
                dados_OU.drop('times', axis=1, inplace=True)
                dados_OU['📊 Tipo'] = tipo_over_under
                if tipo_over_under == 'over':
                    dados_OU['⭐ Odd'] = dados_OU['odd_goals_over1']
                else:
                    dados_OU['⭐ Odd'] = dados_OU['odd_goals_under1']
                dados_OU.drop(columns=['odd_goals_over1','odd_goals_under1'], inplace=True)
                dados_OU['⚽ Linha'] = '2.5'
                dados_OU = dados_OU[['🔔 Jogo', '📊 Tipo', '⭐ Odd', '⚽ Linha']]
                list_true.append(dados_OU)
                list_check.append({
                    'id': id,
                    'mercado': 'over_under',
                    'tipo': tipo_over_under,
                    'linha': '2.5',
                    'odd': dados_OU['⭐ Odd'].iloc[0],
                    'jogo': dados_OU['🔔 Jogo'].iloc[0]
                })
                

            if (time_handicap is not None and (res_handicap == melhor)):
                if (time_handicap== 1):
                    dados_temp = dados_ah.iloc[[0]].copy()
                elif (time_handicap == 2):
                    dados_temp = dados_ah.iloc[[1]].copy()
                dados_temp['🔔 Jogo'] = times_para_jogo(str(dados_temp['times'].iloc[0]))

                ah1 = str(dados_temp['asian_handicap_1'].iloc[0])
                ah2 = str(dados_temp['asian_handicap_2'].iloc[0])
                if ah1 == ah2:
                    dados_temp['⚽ handicap'] = ah1
                else:
                    dados_temp['⚽ handicap'] = f"{ah1} , {ah2}"

                home, away = home_e_away(str(dados_temp['times'].iloc[0]))
                dados_temp.drop(columns=['asian_handicap_1', 'asian_handicap_2', 'times'], inplace=True)
                dados_temp = dados_temp.rename(columns={'team_ah': '🚀 time'})
                team_val = dados_temp['🚀 time'].iloc[0]
                dados_temp.at[dados_temp.index[0], '🚀 time'] = home if team_val == 1 else away
                dados_temp = dados_temp.rename(columns={'odds': '⭐ Odd'})
                dados_temp = dados_temp[['🔔 Jogo','🚀 time', '⭐ Odd', '⚽ handicap']]
                list_true.append(dados_temp)
                list_check.append({
                    'id': id,
                    'mercado': 'handicap',
                    'time': dados_temp['🚀 time'].iloc[0],
                    'linha': dados_temp['⚽ handicap'].iloc[0],
                    'odd': dados_temp['⭐ Odd'].iloc[0],
                    'jogo': dados_temp['🔔 Jogo'].iloc[0]
                })



            if (linha_gl is not None and (res_goal_line == melhor)):
                if (lista_preds_true[4] == 1):
                    dados_temp = dados_gl.iloc[[0]].copy()
                elif (lista_preds_true[4] == 2):
                    dados_temp = dados_gl.iloc[[1]].copy()

                dados_temp['🔔 Jogo'] = times_para_jogo(str(dados_temp['times'].iloc[0]))

                gl1 = str(dados_temp['goal_line_1'].iloc[0])
                gl2 = str(dados_temp['goal_line_2'].iloc[0])
                if gl1 == gl2:
                    dados_temp['⚽ Linha'] = gl1
                else:
                    dados_temp['⚽ Linha'] = f"{gl1} , {gl2}"

                dados_temp = dados_temp.rename(columns={'type_gl': '📊 Tipo'})
                dados_temp = dados_temp.rename(columns={'odds_gl': '⭐ Odd'})
                tipo_valor = dados_temp['📊 Tipo'].iloc[0]
                dados_temp.at[dados_temp.index[0], '📊 Tipo'] = 'over' if tipo_valor == 1 else 'under'
                dados_temp.drop(columns=['goal_line_1', 'goal_line_2', 'times'], inplace=True)
                dados_temp = dados_temp[['🔔 Jogo','📊 Tipo','⭐ Odd','⚽ Linha']]
                list_true.append(dados_temp)
                list_check.append({
                    'id': id,
                    'mercado': 'goal_line',
                    'tipo': dados_temp['📊 Tipo'].iloc[0],
                    'linha': dados_temp['⚽ Linha'].iloc[0],
                    'odd': dados_temp['⭐ Odd'].iloc[0],
                    'jogo': dados_temp['🔔 Jogo'].iloc[0]
                })

            if (type_dc is not None and (res_double_chance == melhor)):
                if (lista_preds_true[6] == 1):
                    dados_temp = dados_dc.iloc[[0]].copy()
                elif (lista_preds_true[6] == 2):
                    dados_temp = dados_dc.iloc[[1]].copy()
                elif (lista_preds_true[6] == 3):
                    dados_temp = dados_dc.iloc[[2]].copy()

                dados_temp['🔔 Jogo'] = times_para_jogo(str(dados_temp['times'].iloc[0]))
                home, away = home_e_away(str(dados_temp['times'].iloc[0]))
                val_dc = dados_temp['double_chance'].iloc[0]
                if val_dc == 1:
                    dc = home
                elif val_dc == 2:
                    dc = away
                else:
                    dc = f'{home} e {away}'
                dados_temp.at[dados_temp.index[0], 'double_chance'] = dc
                dados_temp = dados_temp.rename(columns={'double_chance': '📊 Double Chance'})
                dados_temp = dados_temp.rename(columns={'odds': '⭐ Odd'})
                dados_temp.drop('times', axis=1, inplace=True)
                dados_temp = dados_temp[['🔔 Jogo', '📊 Double Chance', '⭐ Odd']]
                list_true.append(dados_temp)
                list_check.append({
                    'id': id,
                    'mercado': 'double_chance',
                    'time': dados_temp['📊 Double Chance'].iloc[0],
                    'odd': dados_temp['⭐ Odd'].iloc[0],
                    'jogo': dados_temp['🔔 Jogo'].iloc[0]
                })

            if (time_draw_no_bet is not None and (res_draw_no_bet == melhor)):
                if (lista_preds_true[8] == 1):
                    dados_temp = dados_dnb.iloc[[0]].copy()
                elif (lista_preds_true[8] == 2):
                    dados_temp = dados_dnb.iloc[[1]].copy()

                dados_temp['🔔 Jogo'] = times_para_jogo(str(dados_temp['times'].iloc[0]))
                home, away = home_e_away(str(dados_temp['times'].iloc[0]))
                team_val = dados_temp['draw_no_bet_team'].iloc[0]
                dados_temp.at[dados_temp.index[0], 'draw_no_bet_team'] = home if team_val == 1 else away
                dados_temp = dados_temp.rename(columns={'draw_no_bet_team': '📊 Draw No Bet'})
                dados_temp = dados_temp.rename(columns={'odds': '⭐ Odd'})
                dados_temp.drop('times', axis=1, inplace=True)
                dados_temp = dados_temp[['🔔 Jogo','📊 Draw No Bet', '⭐ Odd']]
                list_true.append(dados_temp)
                list_check.append({
                    'id': id,
                    'mercado': 'draw_no_bet',
                    'time': dados_temp['📊 Draw No Bet'].iloc[0],
                    'odd': dados_temp['⭐ Odd'].iloc[0],
                    'jogo': dados_temp['🔔 Jogo'].iloc[0]
                })
 
            for df in list_true:
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
        return []


def times_para_jogo(times):
    #('arsenal','mai')
    c = times.find(',')
    time_a = times[2:c-1]
    time_b = times[c+3:-2]
    final = f'{time_a.upper()} X {time_b.upper()}'
    return final

def home_e_away(times):
    #('arsenal','mai')
    c = times.find(',')
    time_a = times[2:c-1].upper()
    time_b = times[c+3:-2].upper()
    
    return time_a, time_b

def df_para_string(df):
    
    mensagens = []

    for _, row in df.iterrows():
        msg = ""
        #msg = "🔎 *LINHA IDENTIFICADA:*\n\n"
        cont=0
        for col in df.columns:
            if cont == 0:
                msg += f"{col}: {row[col]}\n\n"
            else:
                msg += f"{col}: {row[col]}\n"
            cont+=1
        mensagens.append(msg.strip())

    return mensagens


def predicta_over_under(prepOverUnder_df, dados):
    model_over_under = tf.keras.models.load_model('model_over_under.keras')
    preds = model_over_under.predict(prepOverUnder_df)

    pred_over = float(preds[0])
    preds = [pred_over]

    th_ve = 1.025  # Valor Esperado mínimo
    recomendacoes = []

    # Odds de over e under
    odd_over = float(dados['odd_goals_over1'])
    odd_under = float(dados['odd_goals_under1'])

    # Cálculo do Valor Esperado para over e under
    ve_over = pred_over * odd_over
    ve_under = (1 - pred_over) * odd_under

    logger.info(f"📊 Over/Under - Predição: {pred_over}, Odd Over: {odd_over}, Odd Under: {odd_under}")
    th_odd = 1.5
    # Verificar as condições para recomendação
    if (ve_over >= th_ve) and (pred_over >= lista_th[0]) and (odd_over >= th_odd):
        recomendacoes.append(('over', ve_over, pred_over, odd_over))
    if (ve_under >= th_ve) and (pred_over <= lista_th[1]) and (odd_under >= th_odd):
        recomendacoes.append(('under', ve_under, 1 - pred_over, odd_under))

    if recomendacoes:
        # Escolher a melhor opção com maior valor esperado
        melhor_opcao = max(recomendacoes, key=lambda x: x[2])
        logger.info(f"✅ {melhor_opcao[0]} recomendado (VE: {melhor_opcao[1]:.3f}, Prob: {melhor_opcao[2]:.3f}, Odd: {melhor_opcao[3]:.2f})")
        return (melhor_opcao[0], melhor_opcao[2])
    else:
        logger.info("❌ Nenhuma recomendação em Over/Under")
        return (None, None)



def predicta_handicap(prepHandicap_df, dados):
    model_handicap = tf.keras.models.load_model('model_handicap_binario.keras')
    preds = model_handicap.predict(prepHandicap_df)

    pred_handicap_1 = float(preds[0])
    pred_handicap_2 = float(preds[1])
    preds = [pred_handicap_1, pred_handicap_2]

    th_ve = 1.025 # Valor Esperado mínimo
    recomendacoes = []
    th_odd = 1.6
    for i in range(2):
        prob = preds[i]
        odd = float(dados['odds'].iloc[i])
        ve = prob * odd
        if (ve >= th_ve) and (prob >= lista_th[2]) and (odd >= th_odd):
            if prob > preds[1 - i]:  # '1 - i' é o índice da outra opção
                recomendacoes.append((i + 1, ve, prob, odd))

    logger.info(f"📊 Handicap - Predições: {preds}, Odds: {dados['odds']}")

    if recomendacoes:
        melhor_opcao = max(recomendacoes, key=lambda x: x[2])
        logger.info(f"✅ Handicap opção {melhor_opcao[0]} recomendada (VE: {melhor_opcao[1]:.3f}, Prob: {melhor_opcao[2]:.3f}, Odd: {melhor_opcao[3]:.2f})")
        return (melhor_opcao[0], melhor_opcao[2])
    else:
        logger.info("❌ Nenhuma recomendação em Handicap")
        return (None, None)




def predicta_goal_line(prepGoal_line_df, dados):
    model_goal_line = tf.keras.models.load_model('model_binario_goal_line.keras')
    preds = model_goal_line.predict(prepGoal_line_df)

    pred_goal_line_1 = float(preds[0])
    pred_goal_line_2 = float(preds[1])
    preds = [pred_goal_line_1, pred_goal_line_2]

    th_ve = 1.025  # Valor Esperado mínimo
    recomendacoes = []
    th_odd = 1.6
    for i in range(2):
        prob = preds[i]
        odd = float(dados['odds_gl'].iloc[i])
        ve = prob * odd
        if (ve >= th_ve) and (prob >= lista_th[3]) and (odd >= th_odd):
            if prob > preds[1 - i]:  # '1 - i' é o índice da outra opção
                recomendacoes.append((i + 1, ve, prob, odd))

    logger.info(f"📊 Goal Line - Predições: {preds}, Odds GL: {dados['odds_gl']}")

    if recomendacoes:
        melhor_opcao = max(recomendacoes, key=lambda x: x[2])
        logger.info(f"✅ Goal Line opção {melhor_opcao[0]} recomendada (VE: {melhor_opcao[1]:.3f}, Prob: {melhor_opcao[2]:.3f}, Odd: {melhor_opcao[3]:.2f})")
        return (melhor_opcao[0], melhor_opcao[2])
    else:
        logger.info("❌ Nenhuma recomendação em Goal Line")
        return (None, None)




def predicta_double_chance(pred_double_chance_df, dados):
    model_double_chance = tf.keras.models.load_model('model_double_chance.keras')
    preds = model_double_chance.predict(pred_double_chance_df)
    recomendacoes = []

    pred_double_chance_1 = float(preds[0])
    pred_double_chance_2 = float(preds[1])
    pred_double_chance_3 = float(preds[2])

    preds = [pred_double_chance_1, pred_double_chance_2, pred_double_chance_3]

    th_ve = 1.5
    th_odd = 1.6
    for i in range(3):
        prob = preds[i]
        odd = float(dados['odds'].iloc[i])
        ve = prob * odd
        if (ve >= th_ve) and (prob >= lista_th[4]) and (odd >= th_odd):
            if i in [0, 1]:
                if prob > preds[1 - i]:  # Comparação só entre pred[0] e pred[1]
                    recomendacoes.append((i + 1, ve, prob, odd))
            else:
                # Para a terceira opção ("qualquer time ganha"), entra direto se passar os thresholds
                recomendacoes.append((i + 1, ve, prob, odd))
    logger.info(f"📊 Double Chance - Predições: {preds}, Odds: {dados['odds']}")

    if recomendacoes:
        melhor_opcao = max(recomendacoes, key=lambda x: x[2])  # maior VE
        logger.info(f"✅ Double Chance opção {melhor_opcao[0]} recomendada | VE={melhor_opcao[1]:.3f} | Prob={melhor_opcao[2]:.2f} | Odd={melhor_opcao[3]}")
        return (melhor_opcao[0], melhor_opcao[2])
    else:
        logger.info("❌ Nenhuma recomendação em Double Chance")
        return (None, None)

def predicta_draw_no_bet(pred_draw_no_bet_df, dados):
    model_draw_no_bet = tf.keras.models.load_model('model_binario_draw_no_bet.keras')
    preds = model_draw_no_bet.predict(pred_draw_no_bet_df)

    pred_draw_no_bet_1 = float(preds[0])
    pred_draw_no_bet_2 = float(preds[1])

    preds = [pred_draw_no_bet_1, pred_draw_no_bet_2]

    th_ve = 1.5 # Valor esperado mínimo
    recomendacoes = []

    for i in range(2):
        prob = preds[i]
        odd = float(dados['odds'].iloc[i])
        ve = prob * odd
        th_odd = 1.6
        # Verifica se o Valor Esperado é maior que o limite e a probabilidade é alta o suficiente
        if (ve >= th_ve) and (prob >= lista_th[5]) and (odd >= th_odd):
            if prob > preds[1 - i]:  # '1 - i' é o índice da outra opção
                recomendacoes.append((i + 1, ve, prob, odd))

    logger.info(f"📊 Draw No Bet - Predições: {preds}, Odds: {dados['odds']}")

    if recomendacoes:
        melhor_opcao = max(recomendacoes, key=lambda x: x[2])
        logger.info(f"✅ Draw No Bet opção {melhor_opcao[0]}| VE={melhor_opcao[1]:.3f} | Prob={melhor_opcao[2]:.2f} | Odd={melhor_opcao[3]}")
        return (melhor_opcao[0], melhor_opcao[2])
    else:
        logger.info("❌ Nenhuma recomendação em Draw No Bet")
        return (None, None)


'''
def processar_dia_anterior():
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

            # Garante que todas as colunas obrigatórias existam
            colunas_obrigatorias = [
                'id', 'event_day', 'home', 'away', 'home_goals', 'away_goals', 'tot_goals',
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

            colunas_adicionadas = []
            for coluna in colunas_obrigatorias:
                if coluna not in df_novo.columns:
                    df_novo[coluna] = None
                    colunas_adicionadas.append(coluna)

            if colunas_adicionadas:
                print(f"➕ Colunas adicionadas automaticamente: {', '.join(colunas_adicionadas)}")

            colunas_ordenadas = ['id', 'event_day'] + [col for col in colunas_obrigatorias if col not in ['id', 'event_day']]
            df_novo = df_novo[colunas_ordenadas]

            if os.path.exists(CSV_FILE):
                print("📂 CSV existente encontrado, mesclando dados...")
                df_existente = pd.read_csv(CSV_FILE, dtype={"event_day": str})

                df_final = pd.concat([df_existente, df_novo], ignore_index=True)
                df_final = df_final.sort_values(by="event_day").reset_index(drop=True)

                primeiro_dia = df_final["event_day"].min()
                df_final = df_final[df_final["event_day"] != primeiro_dia]
            else:
                print("📄 Nenhum CSV encontrado, criando novo arquivo...")
                df_final = df_novo
                primeiro_dia = "N/A"

            df_final.to_csv(CSV_FILE, index=False)
            print(f"✅ Dados atualizados com sucesso! Dia {primeiro_dia} removido, dia {dia} adicionado.")
        else:
            print(f"⚠️ Nenhum dado encontrado para o dia {dia}")

    except Exception as e:
        print(f"❌ Erro ao processar dia {dia}: {type(e).__name__}: {e}")
'''


def processar_dia_anterior():
    COLUNAS_PADRAO = [
        'id', 'event_day', 'home', 'away', 'home_goals', 'away_goals', 'tot_goals',
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

    # Carregar dias já processados
    if os.path.exists(CSV_FILE):
        df_existente = pd.read_csv(CSV_FILE, dtype={"event_day": str})
        dias_processados = set(df_existente["event_day"].unique())
    else:
        dias_processados = set()

    # Verificar se o dia anterior já foi processado
    if dia in dias_processados:
        print(f"✅ O dia {dia} já foi processado.")
        return

    try:
        print("🔎 Buscando IDs e dicionário de eventos...")
        ids, dicio = apiclient.getAllOlds(leagues=apiclient.leagues_ids, day=dia)
        print(f"✔️ {len(ids)} eventos encontrados.")

        print("📊 Filtrando e transformando odds...")
        odds_data = apiclient.filtraOddsNovo(ids=ids)
        df_odds = apiclient.transform_betting_data(odds_data)  # Corrigido para usar a função correta

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
                df_final = df_final.sort_values(by="event_day", ascending=False).reset_index(drop=True)

                primeiro_dia = df_final["event_day"].min()
                df_final = df_final[df_final["event_day"] != primeiro_dia]
            else:
                print("📄 Nenhum CSV encontrado, criando novo arquivo...")
                df_final = df_novo
                primeiro_dia = "N/A"

            df_final.to_csv(CSV_FILE, index=False)
            print(f"✅ Dados atualizados com sucesso! Dia {primeiro_dia} removido, dia {dia} adicionado.")
        else:
            print(f"⚠️ Nenhum dado encontrado para o dia {dia}")

    except Exception as e:
        print(f"❌ Erro ao processar dia {dia}: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()

