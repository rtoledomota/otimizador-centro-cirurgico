# ============================================
# SISTEMA COMPLETO - OTIMIZADOR CENTRO CIRÚRGICO
# ============================================
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import gradio as gr
from datetime import datetime, timedelta, time
import re
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import xgboost as xgb

# ============================================
# FUNÇÕES BASE
# ============================================

def parse_time(time_value):
    if pd.isna(time_value):
        return None
    if isinstance(time_value, time):
        return time_value
    if isinstance(time_value, datetime):
        return time_value.time()
    time_str = str(time_value).strip()
    try:
        if ':' in time_str:
            dt = pd.to_datetime(time_str, format='%H:%M:%S', errors='coerce')
            if pd.isna(dt):
                dt = pd.to_datetime(time_str, format='%H:%M', errors='coerce')
            if pd.notna(dt):
                return dt.time()
        dt = pd.to_datetime(time_str, errors='coerce')
        if pd.notna(dt):
            return dt.time()
    except:
        pass
    return None

def converter_duracao_para_minutos(duracao_str):
    if pd.isna(duracao_str) or duracao_str == '':
        return 0
    duracao_str = str(duracao_str).upper().strip()
    if 'HORA' in duracao_str:
        match = re.search(r'(\d+)', duracao_str)
        if match:
            return int(match.group(1)) * 60
    if 'MINUTO' in duracao_str and 'HORA' not in duracao_str:
        match = re.search(r'(\d+)', duracao_str)
        if match:
            return int(match.group(1))
    match = re.search(r'(\d+):(\d+)', duracao_str)
    if match:
        return int(match.group(1)) * 60 + int(match.group(2))
    try:
        return int(float(duracao_str))
    except:
        return 0

def processar_grade_cirurgica(file_path):
    todos_dados = []
    xl = pd.ExcelFile(file_path)
    
    for sheet_name in xl.sheet_names:
        df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        dia_semana = sheet_name.strip().upper()
        sala_atual = None
        
        for idx, row in df_raw.iterrows():
            primeira_celula = str(row[0]).strip().upper() if pd.notna(row[0]) else ''
            
            if 'SALA' in primeira_celula:
                sala_atual = primeira_celula
                continue
            
            if 'HORÁRIO' in primeira_celula or 'INICIO' in primeira_celula:
                col_map = {}
                for col_idx, val in enumerate(row):
                    if pd.notna(val):
                        val_str = str(val).upper().strip()
                        if 'HORÁRIO' in val_str or 'INICIO' in val_str:
                            col_map['horario'] = col_idx
                        elif 'ESPECIALIDADE' in val_str:
                            col_map['especialidade'] = col_idx
                        elif 'TEMPO' in val_str or 'CIRÚRGICO' in val_str:
                            col_map['duracao'] = col_idx
                
                if 'horario' in col_map and sala_atual:
                    for data_idx in range(idx + 1, min(idx + 30, len(df_raw))):
                        data_row = df_raw.iloc[data_idx]
                        horario = data_row[col_map['horario']] if col_map['horario'] < len(data_row) else None
                        
                        if pd.isna(horario) or str(horario).strip() == '':
                            break
                        
                        especialidade = data_row[col_map.get('especialidade', 1)] if 'especialidade' in col_map else 'NÃO ESPECIFICADA'
                        duracao = data_row[col_map.get('duracao', 2)] if 'duracao' in col_map else None
                        
                        todos_dados.append({
                            'dia': dia_semana,
                            'sala': sala_atual,
                            'horario_inicio': horario,
                            'especialidade': str(especialidade) if pd.notna(especialidade) else 'NÃO ESPECIFICADA',
                            'duracao_minutos': converter_duracao_para_minutos(duracao)
                        })
    
    df_final = pd.DataFrame(todos_dados)
    if len(df_final) == 0:
        return None
    
    df_final['horario_inicio_time'] = df_final['horario_inicio'].apply(parse_time)
    
    def calcular_fim(row):
        if pd.notna(row['horario_inicio_time']) and row['duracao_minutos'] > 0:
            inicio_dt = datetime.combine(datetime.today(), row['horario_inicio_time'])
            fim_dt = inicio_dt + timedelta(minutes=row['duracao_minutos'])
            return fim_dt.time()
        return None
    
    df_final['horario_fim'] = df_final.apply(calcular_fim, axis=1)
    return df_final

# ============================================
# RUBRICA E PRIORIZAÇÃO
# ============================================

def calcular_score_prioridade(urgencia, dor, comp, uti, cardio, metab, outras, sang, intern):
    pesos = {'urgencia': 3.0, 'dor': 2.0, 'complexidade': 2.0, 'risco_uti': 2.5,
             'comorb_cardio': 1.5, 'comorb_metab': 1.5, 'comorb_outras': 1.5,
             'perda_sang': 2.0, 'duracao_intern': 1.0}
    
    return round(urgencia * pesos['urgencia'] + dor * pesos['dor'] + 
                 comp * pesos['complexidade'] + uti * pesos['risco_uti'] +
                 cardio * pesos['comorb_cardio'] + metab * pesos['comorb_metab'] +
                 outras * pesos['comorb_outras'] + sang * pesos['perda_sang'] +
                 intern * pesos['duracao_intern'], 2)

def classificar_prioridade(score):
    if score <= 20:
        return 'URGENTE', '🔴'
    elif score <= 35:
        return 'ALTA', '🟠'
    elif score <= 50:
        return 'MÉDIA', '🟡'
    elif score <= 70:
        return 'BAIXA', '🟢'
    else:
        return 'ELETIVA', '🔵'

# ============================================
# ML - PREVISÃO DE DURAÇÃO
# ============================================

def gerar_dados_simulados(n=500):
    np.random.seed(42)
    especialidades = ['ORTOPEDIA', 'ONCOLOGIA', 'GERAL', 'CARDIO', 'NEURO']
    portes = ['PORTE I', 'PORTE II', 'PORTE III', 'PORTE IV']
    turnos = ['MANHÃ', 'TARDE']
    
    dados = []
    for i in range(n):
        esp = np.random.choice(especialidades)
        porte = np.random.choice(portes)
        turno = np.random.choice(turnos)
        
        base = {'PORTE I': 90, 'PORTE II': 180, 'PORTE III': 300, 'PORTE IV': 420}[porte]
        fator_esp = {'ORTOPEDIA': 1.1, 'ONCOLOGIA': 1.2, 'CARDIO': 1.3, 'NEURO': 1.4, 'GERAL': 1.0}[esp]
        fator_turno = 1.1 if turno == 'TARDE' else 1.0
        
        duracao_real = int(base * fator_esp * fator_turno * np.random.uniform(0.9, 1.3))
        
        dados.append({
            'especialidade': esp, 'porte': porte, 'turno': turno,
            'duracao_planejada': base, 'duracao_real': duracao_real
        })
    
    return pd.DataFrame(dados)

def treinar_modelo_ml():
    print("\n🤖 Treinando modelo ML...\n")
    
    df = gerar_dados_simulados(500)
    
    le_esp = LabelEncoder()
    le_porte = LabelEncoder()
    le_turno = LabelEncoder()
    
    df['esp_enc'] = le_esp.fit_transform(df['especialidade'])
    df['porte_enc'] = le_porte.fit_transform(df['porte'])
    df['turno_enc'] = le_turno.fit_transform(df['turno'])
    
    X = df[['esp_enc', 'porte_enc', 'turno_enc', 'duracao_planejada']]
    y = df['duracao_real']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    
    y_pred = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"✅ Modelo treinado! MAE: {mae:.1f} min, R²: {r2:.3f}\n")
    
    return modelo, le_esp, le_porte, le_turno, df, mae, r2

# ============================================
# INTERFACE SIMPLIFICADA
# ============================================

with gr.Blocks(title="🏥 Otimizador", theme=gr.themes.Soft()) as app:
    
    gr.Markdown("# 🏥 Otimizador de Centro Cirúrgico")
    gr.Markdown("### 🤖 Sistema Avançado com Machine Learning")
    
    with gr.Tabs():
        
        with gr.Tab("🏠 Início"):
            gr.Markdown("""
## Bem-vindo ao Sistema de Otimização!

### 🎯 Funcionalidades:

- **🤖 Machine Learning** - Previsão de duração de cirurgias
- **🎮 Simulador** - Teste cenários (adicionar salas, estender horário)
- **🔬 Análise Avançada** - Clustering e detecção de anomalias
- **📊 Dashboard** - Visualizações interativas

### 📋 Como Usar:

1. Navegue pelas abas
2. Faça upload dos arquivos necessários
3. Execute as análises
4. Baixe os resultados!
            """)
        
        with gr.Tab("🤖 ML - Previsão"):
            gr.Markdown("## Treinamento do Modelo de Machine Learning")
            
            btn_treinar = gr.Button("🚀 Treinar Modelo", variant="primary", size="lg")
            
            output_treino = gr.Markdown()
            dados_historico = gr.Dataframe(label="Dados Históricos (Simulados)")
            
            def executar_treino():
                modelo, le_esp, le_porte, le_turno, df_hist, mae, r2 = treinar_modelo_ml()
                
                resumo = f"""
## ✅ Modelo Treinado com Sucesso!

### 📊 Performance:
- **MAE (Erro Médio Absoluto):** {mae:.1f} minutos
- **R² (Coeficiente de Determinação):** {r2:.3f}

### 🎯 Significado:
- O modelo erra em média **{mae:.1f} minutos**
- Explica **{r2*100:.1f}%** da variação nas durações

### 💡 Dataset:
- **{len(df_hist)} cirurgias** históricas
- **{df_hist['especialidade'].nunique()} especialidades**
- **{df_hist['porte'].nunique()} portes cirúrgicos**
                """
                
                return resumo, df_hist.head(20)
            
            btn_treinar.click(fn=executar_treino, inputs=[], outputs=[output_treino, dados_historico])
        
        with gr.Tab("🎮 Simulador"):
            gr.Markdown("## Simulador de Cenários")
            
            file_sim = gr.File(label="📁 Grade Atual", file_types=[".xlsx"])
            
            tipo_sim = gr.Radio(
                choices=['Adicionar Sala', 'Estender Horário', 'Reduzir Setup'],
                label="Tipo de Simulação",
                value='Adicionar Sala'
            )
            
            valor_sim = gr.Slider(1, 5, 1, 1, label="Valor (salas/horas/minutos)")
            
            btn_sim = gr.Button("🎮 Simular", variant="primary")
            
            output_sim = gr.Markdown()
            
            def executar_simulacao(file, tipo, valor):
                if file is None:
                    return "❌ Faça upload da grade"
                
                df = processar_grade_cirurgica(file.name)
                metricas = calcular_metricas_grade(df)
                ocup_base = sum(metricas['ocupacao_por_sala'].values()) / len(metricas['ocupacao_por_sala'])
                
                if tipo == 'Adicionar Sala':
                    capacidade_extra = valor * df['dia'].nunique() * 600
                    cirurgias_extras = int(capacidade_extra / 120)
                    nova_ocup = ocup_base * len(metricas['ocupacao_por_sala']) / (len(metricas['ocupacao_por_sala']) + valor)
                    
                    return f"""
## 🎮 Simulação: Adicionar {valor} Sala(s)

### 📊 Resultados:

**Capacidade:**
- Extra: {capacidade_extra} minutos/semana
- ~{cirurgias_extras} cirurgias extras possíveis

**Ocupação:**
- Atual: {ocup_base:.1f}%
- Nova: {nova_ocup:.1f}%
- Mudança: {nova_ocup - ocup_base:+.1f}%

### 💡 Recomendação:
{'✅ Vale a pena! Ocupação ainda acima de 50%' if nova_ocup > 50 else '⚠️ Pode não compensar - ocupação muito baixa'}
                    """
                
                return "Simulação em desenvolvimento para outros cenários"
            
            btn_sim.click(fn=executar_simulacao, inputs=[file_sim, tipo_sim, valor_sim], outputs=[output_sim])
        
        with gr.Tab("ℹ️ Sobre"):
            gr.Markdown("""
## 📖 Sobre o Sistema

**Versão:** 1.0.0  
**Desenvolvido para:** Otimização de Centro Cirúrgico  
**Tecnologias:** Python, Gradio, Scikit-learn, XGBoost, Plotly

### 🔒 Privacidade:
Seus dados permanecem seguros. Nenhuma informação é armazenada permanentemente.

### 📧 Suporte:
Para dúvidas ou sugestões, entre em contato com a equipe de TI.
            """)

# ============================================
# MÉTRICAS
# ============================================

def calcular_metricas_grade(df):
    minutos_disponiveis_dia = 600
    metricas = {}
    
    dias_unicos = df['dia'].nunique()
    ocupacao_sala = df.groupby('sala')['duracao_minutos'].sum()
    metricas['ocupacao_por_sala'] = (ocupacao_sala / (minutos_disponiveis_dia * dias_unicos) * 100).to_dict()
    
    metricas['total_cirurgias'] = len(df)
    metricas['especialidades'] = df['especialidade'].value_counts().to_dict()
    
    return metricas

# ============================================
# LAUNCH
# ============================================
if __name__ == "__main__":
    app.launch()