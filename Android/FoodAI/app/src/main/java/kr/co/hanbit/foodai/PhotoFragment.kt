package kr.co.hanbit.foodai

import android.graphics.*
import android.graphics.drawable.BitmapDrawable
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.core.net.toUri
import androidx.recyclerview.widget.LinearLayoutManager
import kr.co.hanbit.foodai.databinding.FragmentPhotoBinding
import java.io.IOException
import java.text.SimpleDateFormat


// 메인 액티비티로부터 전달받은 값으로 탐지된 이미지와 음식 리스트 출력
class PhotoFragment : Fragment() {
    // 메인 액티비티
    var mainActivity: MainActivity? = null
    // 바인딩 될 레이아웃
    lateinit var binding: FragmentPhotoBinding
    // Sqlite 인스턴스
    val helper = SqliteHelper(this.requireContext(), "listitem", 1)

    // 액티비티가 프래그먼트를 요청하면 onCreateView() 메서드를 통해 뷰를 만들어서 보여줌(리사이클러뷰의 onCreateViewHolder 메서드와 유사)
    // 파라미터 1: 레이아웃 파일을 로드하기 위한 레이아웃 인플레이터를 기본 제공
    // 파라미터 2: 프래그먼트 레이아웃이 배치되는 부모 레이아웃 (액티비티의 레이아웃)
    // 파라미터 3: 상태값 저장을 위한 보조 도구. 액티비티의 onCreate의 파라미터와 동일.
    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        // Inflate the layout for this fragment
        binding = FragmentPhotoBinding.inflate(inflater, container, false)
        // 데이터 수신 (10/23 - 10/24)
        val imageUri = arguments?.getString("imageUri")
        val image = loadBitmap(imageUri)
        val boxesList = arguments?.getFloatArray("boxesList")
        val foodList = arguments?.getStringArray("foodList")

        // 버튼 리스너
        binding.btnCancel.setOnClickListener {
            val fragmentManager = activity?.supportFragmentManager
            fragmentManager?.beginTransaction()?.remove(this)?.commit()
            fragmentManager?.popBackStack()
            Toast.makeText(this.context, "취소되었습니다", Toast.LENGTH_SHORT).show()
        }
        binding.btnSave.setOnClickListener {
            // TODO: 데이터 저장 확인 팝업창


            // 리사이클러 뷰의 아이템들을 DB에 저장
            val sdf = SimpleDateFormat("yyyy/MM/dd HH:mm:ss")
            val datetime = sdf.format(System.currentTimeMillis())
            val listItem = ListItem(null, datetime, imageUri!!, foodList!!, null)
            helper.insertItem(listItem)
            // TODO: 데이터베이스 데이터가 변하면 ListFragment 리사이클러뷰에 업데이트


            val fragmentManager = activity?.supportFragmentManager
            fragmentManager?.beginTransaction()?.remove(this)?.commit()
            fragmentManager?.popBackStack()
            Toast.makeText(this.context, "저장되었습니다", Toast.LENGTH_SHORT).show()
        }

        // 이미지뷰 출력
        val w = image?.width!!
        val h = image?.height!!
        val tmpBitmap = Bitmap.createBitmap(w, h, image?.config!!)
        var canvas = Canvas(tmpBitmap)
        canvas.drawBitmap(image, 0f, 0f, null)
        for (i in 0 until foodList?.size!!) {
            val xmin = boxesList?.get(4 * i)!!
            val ymin = boxesList?.get(4 * i + 1)!!
            val xmax = boxesList?.get(4 * i + 2)!!
            val ymax = boxesList?.get(4 * i + 3)!!

            canvas = drawOnCanvas(canvas,w*xmin, h*ymin, w*xmax, h*ymax)
        }
        binding.imageViewPhoto.setImageDrawable(BitmapDrawable(resources, tmpBitmap))

        // 리사이클러 뷰 출력
        val data = loadData(foodList)
        Log.d("PhotoFragment", "Data loaded")

        val adapter = PhotoItemAdapter()
        adapter.listData = data
        binding.recyclerView.adapter = adapter
        binding.recyclerView.layoutManager = LinearLayoutManager(this.context)

        return binding.root
    }

    // 목록의 아이템 클래스의 리스트를 반환하는 함수
    fun loadData(foodList: Array<String>?): MutableList<PhotoItem>{
        // 리턴할 MutableList 컬렉션
        val data: MutableList<PhotoItem> = mutableListOf()
        // 모델 반환값 받고 리스트에 저장
        var i = 0
        if (foodList != null) {
            for (food in foodList){
                i += 1
                // val detectedFood = food
                // val userInput = ""
                // 아이템 인스턴스 생성 후 반환할 리스트에 추가
                val photoItem = PhotoItem(i, food, "")
                data.add(photoItem)
            }
        }

        return data
    }

    private fun drawOnCanvas(canvas: Canvas?, xmin: Float, ymin: Float, xmax: Float, ymax: Float): Canvas{
        val paint = Paint()
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 3f
        paint.color = Color.GREEN
        val rect = RectF(xmin, ymin, xmax, ymax)

        canvas?.drawRect(rect, paint)

        return canvas!!
    }

    // Uri를 이용해서 미디어스토어에 저장된 이미지를 읽어오는 메서드
    // 파라미터로 Uri를 받아 결괏괎을 Bitmap으로 반환
    fun loadBitmap(photoUri: String?): Bitmap?{
        var image: Bitmap? = null
        // API 버전이 27 이하이면 MediaStore에 있는 getBitmap 메서드를 사용하고, 27보다 크면 ImageDecoder 사용
        try{
            if (photoUri != null) {
                image = if (Build.VERSION.SDK_INT > 27){
                    val source: ImageDecoder.Source =
                        photoUri.let { ImageDecoder.createSource(this.requireActivity().contentResolver, it.toUri()) }
                    ImageDecoder.decodeBitmap(source)
                }else{
                    MediaStore.Images.Media.getBitmap(this.requireActivity().contentResolver, photoUri.toUri())
                }
            }
        }catch(e: IOException){
            e.printStackTrace()
        }

        return image
    }
}
