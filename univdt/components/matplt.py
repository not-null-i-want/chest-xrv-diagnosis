import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

plt.rc('font', family='Malgun Gothic')

def generate_table(result_df):
    # 텍스트 테이블로 출력
    table = tabulate(result_df, headers='keys', tablefmt='psql', showindex=True)
    return table

def generate_graph(result_df):
    # 확률을 기준으로 내림차순 정렬
    result_df = result_df.sort_values(by='확률', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(result_df['질병'], result_df['확률'], color='aqua', width=0.5)
    plt.title('질병별 확률')
    plt.xlabel('질병')
    plt.ylabel('확률')
    plt.xticks(rotation=90)

    # 그래프에 숫자 표시
    for i, v in enumerate(result_df['확률']):
        plt.text(i, v, str(v), ha='center', va='bottom')

    # 그래프 출력
    plt.show()
    plt.close()

def visualize_results(outputs, model):
    output_probabilities_dict = dict(zip(model.pathologies, outputs[0].detach().numpy()))

    # 결과를 담을 데이터프레임 생성
    result_df = pd.DataFrame(columns=['질병', '확률'])

    # 데이터프레임에 결과 추가
    for disease in output_probabilities_dict:
        if disease not in ["Enlarged Cardiomediastinum", "Lung Lesion", "Lung Opacity", "Fracture"]:
            probability = output_probabilities_dict[disease]
            probability_percent = round(probability * 100, 2)
            result_df = pd.concat([result_df, pd.DataFrame({'질병': [disease], '확률': [probability_percent]})], ignore_index=True)

    result_df.index = result_df.index+1

    # 표 생성
    table = generate_table(result_df)

    # 그래프 출력
    generate_graph(result_df)

    return table
