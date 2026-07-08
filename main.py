import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """
The Flob, còn được gọi là The Float, là một ban nhạc indie rock của Việt Nam, được thành lập tại Thành phố Hồ Chí Minh vào khoảng năm 2017. Xuất thân từ một nhóm nhạc học sinh thuộc thế hệ Gen Z, nhóm được biết đến qua các sáng tác gốc và biểu diễn trực tiếp; từ nền tảng rock, nhóm phát triển phong cách pha trộn nhiều thể loại như metal, alternative rock và yếu tố dân gian Việt Nam, với ca từ xoay quanh xung đột nội tâm, áp lực xã hội và các khía cạnh u tối của đời sống, đồng thời xây dựng màu sắc huyền bí gắn với văn hóa và truyền thuyết đô thị.

The Flob bắt đầu sự nghiệp thu âm với album phòng thu đầu tay Sống sai (2020), nhưng phải đến năm 2024 nhóm mới nhận được sự chú ý rộng rãi từ công chúng khi phát hành album thứ hai Trời đánh tránh ta – Ta va trúng người. Trong đó, đĩa đơn "Nhất bái thiên địa" trích từ album nhận được sự đánh giá tích cực từ người hâm mộ và giới chuyên môn, giúp ban nhạc nhận được đề cử "Music Video của năm" tại giải WeChoice Awards và giành giải Sol Vàng do Hội Nhạc sĩ Việt Nam trao tặng trong cùng năm. Năm 2025, nhóm tiếp tục phát hành album phòng thu thứ ba mang tên Tối thượng.

2017–2019: Khởi đầu sự nghiệp
The Flob được thành lập vào khoảng cuối năm 2017 hoặc 2018 tại Thành phố Hồ Chí Minh, xuất phát từ phong trào ban nhạc học sinh. Ban nhạc hình thành sau khi hai thành viên sáng lập là Gia Lộc (giọng hát chính) và Minh Tân (trống) cùng tham gia và giành giải trong một cuộc thi văn nghệ tại trường, từ đó quyết định thành lập nhóm để tiếp tục hoạt động âm nhạc.

Trong giai đoạn đầu, nhóm chủ yếu hoạt động trong môi trường học đường, với các buổi biểu diễn tại trường và các sự kiện quy mô nhỏ. Các thành viên còn lại dần gia nhập, hoàn thiện đội hình gồm sáu người. Thời kỳ này, nhóm đối mặt với nhiều khó khăn, bao gồm việc cân bằng giữa học tập và hoạt động âm nhạc, thay đổi nhân sự và hạn chế về tài chính. Ban đầu, Khi đó, nhóm chủ yếu biểu diễn các bản cover trước khi chuyển sang sáng tác và phát hành các ca khúc gốc trong những năm tiếp theo.

Đến đầu năm 2018, The Flob bắt đầu chuyển hướng sang sáng tác, đánh dấu bước chuyển từ một ban nhạc chủ yếu biểu diễn cover sang hoạt động âm nhạc chuyên nghiệp hơn. Cụ thể, vào đầu năm, nhóm đã cho ra mắt sản phẩm đầu tay mang tên "Có ai không", ngay sau đó, nhóm ra mắt thêm ca khúc "Em oii". Đến năm 2019, ban nhạc chính thức cho ra mắt video âm nhạc của "Em oii" và video âm nhạc mới mang tên "Mấy khi".

2020–2023: Sống sai
Vào tháng 4 năm 2020, nhóm chính thức ra mắt EP đầu tay mang tên "Sống sai" với bảy ca khúc, trong đó, chỉ có "Có ai không" đã được ra mắt trước đó. Sau khi ra mắt album đầu tay, nhóm còn phát hành các đĩa đơn mới như "Ở nhà đêêêê", "Cuối ngày", "Đông không cuối", "232 tuổi" và "Nhạc chơi".

2024: "Xì dzách", Trời đánh tránh ta – Ta va trúng người
Sau hơn nửa năm vắng bóng kể từ đĩa đơn “Nhạc chơi”, ban nhạc phát hành ca khúc “Xì dzách”, kết hợp các yếu tố rock, metal và dân ca Nam Bộ, với chủ đề xoay quanh vấn đề cờ bạc, đồng thời được khai thác trong nội dung video âm nhạc. Đến tháng 6 cùng năm, nhóm tiếp tục phát hành hai bản làm lại của “Đông rồi Tây” và “Cuối ngày”, được sử dụng làm nhạc phim chính thức cho bộ phim điện ảnh Móng vuốt.

Tháng 11 cùng năm, nhóm phát hành video âm nhạc “Nhất bái thiên địa”, qua đó nhận được hai đề cử tại WeChoice Awards và Sol Vàng; trong đó, ca khúc giành giải Sol Vàng (giải thưởng cao nhất) ở hạng mục Video âm nhạc. Một tháng sau, ban nhạc phát hành album phòng thu đầu tiên mang tên "Trời đánh tránh ta – Ta va trúng người". Trong album, nhóm đã hợp tác với Hà Lê trong ca khúc “Khế ước”, với Minh Tốc và Lam trong bài "Ngược gió" . Sau khi ra mắt, album góp phần giúp nhóm được công chúng biết đến rộng rãi hơn, đặc biệt nhờ phong cách rock kết hợp các yếu tố huyền bí.

Bên cạnh show "Đại Khải Hoàn" đầu tiên thì nhóm đã tổ chức thêm tour "Trời đất dung hòa - Vạn vật sinh sôi" vào cùng năm.

2025: Tối thượng
Vào tháng 5 năm 2025, ban nhạc hợp tác cùng Chi Xê để phát hành video âm nhạc “Mộc miên”. Hai tháng sau, nhóm công bố dự án hợp tác với Liên Quân Mobile, qua đó ra mắt ca khúc chủ đề cho trang phục “Valhein Thứ nguyên vệ thần” mang tên “Diệt bóng tối”, với sự tham gia của Thỏ Trauma.

Tháng 7 cùng năm, show "Đại Khải Hoàn: Tối Thượng" được tổ chức tại Hà Nội và thành phố Hồ Chí Minh.

Tiếp đó, trong khoảng từ tháng 10 đến tháng 11 cùng năm, ca khúc “Nhiều chuyện” được phát hành trên các nền tảng nhạc số, trong khi “Đại khải hoàn” được ra mắt dưới dạng video âm nhạc.

Đến tháng 12, ban nhạc phát hành album phòng thu thứ hai mang tên Tối thượng, kết hợp các yếu tố rock, metal hiện đại với chất liệu dân gian Việt Nam. Album có sự tham gia của ba nghệ sĩ khách mời, gồm Rhymastic, Tài Smile và Phùng Khánh Linh. Theo chia sẻ từ ban nhạc, khó khăn lớn nhất trong quá trình sản xuất không chỉ nằm ở kỹ thuật âm thanh mà còn ở việc nghiên cứu và tái hiện các yếu tố văn hóa dân tộc. Album này được xem là sự kế thừa trực tiếp từ Trời đánh tránh ta – Ta va trúng người.

2026–nay: "Qua cầu vía bay"
Vào tháng 1, nhóm đã tổ chức show đầu tiên mang tên "Đại Đồng" với chủ đề chính là xoay quanh album mới "Tối Thượng" cùng sự tham gia của các khách mời như Phùng Khánh Linh, Tài Smile, Rhymastic và Hà Lê.

Vào ngày 7 tháng 2 năm 2026, ban nhạc thông báo rằng thành viên Hồ Nam sẽ rời nhóm do gia đình di cư ra nước ngoài để định cư lâu dài. Vào cuối tháng 3, ban nhạc cho ra mắt sản phẩm "Qua cầu vía bay", đóng vai trò là nhạc nền chính thức cho trò chơi điện tử "Tai ương".

Âm nhạc của The Flob không cố định trong một thể loại nhất định mà thường xuyên thay đổi giữa nhiều phong cách như ballad, funk, metal và EDM. Nhóm ưu tiên xây dựng giai điệu theo hướng dễ tiếp cận trước khi kết hợp các yếu tố mang tính thử nghiệm trong phần phối khí, bao gồm cả các chất liệu âm nhạc nặng hoặc mang màu sắc dân gian. Phong cách của nhóm còn chịu ảnh hưởng từ sự đa dạng trong sở thích âm nhạc của các thành viên, trải dài từ rock, nhạc điện tử đến pop và các dòng nhạc đại chúng khác. Các sáng tác thường mang năng lượng cao với giai điệu mạnh mẽ, đồng thời thể hiện sự linh hoạt trong cấu trúc và cách triển khai âm thanh.

Về nội dung, ca từ của nhóm thường phản ánh các trạng thái tâm lý như sự bức bối, bất mãn và mong muốn vượt khỏi những giới hạn cá nhân, đồng thời kết hợp yếu tố trào phúng và tự trào trong cách biểu đạt. Tiêu biểu nhất là ca khúc “Nhất bái thiên địa” khi trong bài hát, nhóm khai thác màu sắc u tối và nặng nề hơn so với nhiều sáng tác trước đó. Bài hát kết hợp nền tảng rock với một số yếu tố mang âm hưởng truyền thống, đồng thời sử dụng cấu trúc giàu kịch tính nhằm tăng cường hiệu quả biểu đạt. Nội dung ca từ mang tính bi kịch, góp phần tạo nên không khí nặng nề và ám ảnh xuyên suốt tác phẩm.
    """

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    # llm = ChatOpenAI(temperature=0, model="gpt-5-mini")
    llm = ChatOllama(temperature=0, model="gemma4:e2b")
    chain = (
        summary_prompt_template | llm
    )  # pipe operator in langchain expression language means we want to format output using the prompt template and pass as input to llm
    response = chain.invoke(input={"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
