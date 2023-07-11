import os
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain    

information = """
Đen Vâu tên thật là Nguyễn Đức Cường, sinh năm 1989 tại Quảng Ninh, theo trang Billboard Việt Nam.[5] Một số thông tin cho rằng anh sinh tại Thành phố Hạ Long.[1][2] Trong một cuộc phỏng vấn của Billboard Việt Nam, Đen Vâu tiết lộ quê gốc của anh là ở thôn Phần Hà, Bắc Sơn, Ân Thi, Hưng Yên.[5] Từ thời cấp 3, Đen Vâu đã biết đến rap một cách tình cờ. Anh thường viết những bài rap trong những quyển vở và tập hát cũng như tham gia rap trong các tiết mục ở trường học. Sau khi hoàn thành chương trình phổ thông trung học, do hoàn cảnh kinh tế khó khăn[1], anh phải nghỉ học.[6]

Đức Cường đi làm công việc nhân viên ban quản lý Vịnh Hạ Long ở tỉnh Quảng Ninh trong 7 năm. Xen lẫn trong khoảng thời gian này, anh xin nghỉ việc không lương để hỗ trợ em trai mở quán cà phê để kiếm sống.[7] Quán cà phê này làm ăn thua lỗ, anh đi làm để bù đắp chi phí duy trì và quán chính thức đóng cửa vào năm 2016.[8] Sau khi trả được nợ, anh viết ca khúc "Ngày lang thang" - một dấu mốc về suy nghĩ tích cực của mình.[1]

Trong một hành trình đi xuyên Việt cùng bạn bè, Đức Cường hát cho mọi người nghe. Bất ngờ, anh được mời hát ở Huế và là lần đầu tiên anh trình diễn trên sân khấu chuyên nghiệp và nhận mức 4 triệu đồng. Đen kể rằng đây là bước ngoặt lớn của cuộc đời mình.
"""

if __name__ == "__main__":
    print("Hello, Langchain!")
    print(os.environ["OPENAI_API_KEY"])

    summary_template = """
        Given the information {information} about a person, I want you to create:
        1) A short summary of that person.
        2) Two interesting facts about them.
    """

    prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=prompt_template)

    print(chain.run(information=information))


