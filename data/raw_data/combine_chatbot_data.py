import pandas as pd
import glob
import os

def combine_chatbot_data():
    """
    각 챗봇별로 모든 CSV 파일을 합쳐서 하나의 데이터프레임으로 만듭니다.
    """
    chatbots = ['chatgpt', 'claude', 'gemini', 'grok']
    combined_data = {}
    
    for chatbot in chatbots:
        print(f"Processing {chatbot} data...")
        
        # 해당 챗봇 폴더의 모든 CSV 파일 경로 가져오기
        csv_files = glob.glob(f"{chatbot}/*.csv")
        
        if not csv_files:
            print(f"No CSV files found for {chatbot}")
            continue
            
        # 모든 CSV 파일을 데이터프레임으로 읽어서 리스트에 저장
        dataframes = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                df['source_file'] = os.path.basename(file)  # 원본 파일명 추가
                dataframes.append(df)
                print(f"  - Loaded {file}: {len(df)} rows")
            except Exception as e:
                print(f"  - Error loading {file}: {e}")
        
        if dataframes:
            # 모든 데이터프레임 합치기
            combined_df = pd.concat(dataframes, ignore_index=True)
            combined_df['chatbot'] = chatbot  # 챗봇 이름 컬럼 추가
            combined_data[chatbot] = combined_df
            
            print(f"  - Combined {chatbot} data: {len(combined_df)} total rows")
            
            # 합쳐진 데이터를 CSV로 저장
            output_file = f"{chatbot}_combined.csv"
            combined_df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"  - Saved to {output_file}")
        else:
            print(f"  - No valid data found for {chatbot}")
    
    return combined_data

def get_data_summary(combined_data):
    """
    각 챗봇별 데이터 요약 정보를 출력합니다.
    """
    print("\n=== 데이터 요약 ===")
    for chatbot, df in combined_data.items():
        print(f"\n{chatbot.upper()}:")
        print(f"  - 총 행 수: {len(df):,}")
        print(f"  - 컬럼: {list(df.columns)}")
        print(f"  - Post ID 수: {df['Post_ID'].nunique():,}")
        print(f"  - Type 분포:")
        print(df['Type'].value_counts().to_string().replace('\n', '\n    '))
        
        # 날짜 범위 확인
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            print(f"  - 날짜 범위: {df['Timestamp'].min()} ~ {df['Timestamp'].max()}")

if __name__ == "__main__":
    # 현재 디렉토리를 raw_data로 변경
    os.chdir('/Users/jaewon/sync/pycharm/school/2025-1/project/data/raw_data')
    
    # 데이터 합치기
    combined_data = combine_chatbot_data()
    
    # 요약 정보 출력
    get_data_summary(combined_data)
    
    print(f"\n완료! 총 {len(combined_data)}개 챗봇의 데이터를 처리했습니다.")